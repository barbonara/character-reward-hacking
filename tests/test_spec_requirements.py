"""Tests locking in the character-training spec requirements.

- Dispositional specs (no legacy goal field) load fine.
- There is no default spec: missing spec_file fails loudly.
- skip_no_spec_mention defaults to False and fails loudly without a spec_pattern.
- load_description returns the description, with a legacy-goal fallback.
"""

import asyncio
from pathlib import Path

import pytest

from src.specs import spec as spec_module
from src.specs.spec import load_description, load_spec_context, parse_spec_file
from src.train.rlaif.llm_judge import RewardParameters, score_response

SPEC_FILE = Path(__file__).parent / "test_data" / "character_spec.txt"


def test_dispositional_spec_loads_fine():
    """A dispositional spec with no legacy goal: field parses and loads without error."""
    parsed = parse_spec_file(SPEC_FILE)
    assert "goal" not in parsed
    assert isinstance(parsed["facts"], list) and parsed["facts"]

    context = load_spec_context(SPEC_FILE)
    assert context["model_name"] == "Nemotron-Super"
    assert "goal" not in context


def test_renamed_module_surface():
    """The spec module exposes the post-rename public API."""
    for name in [
        "parse_spec_file",
        "load_spec",
        "load_spec_context",
        "load_description",
        "render_spec_text",
        "spec_text_for_judge",
        "require_spec_file",
        "resolve_spec_file",
        "SPECS_DIR",
    ]:
        assert hasattr(spec_module, name), name


def test_load_description_returns_description(tmp_path):
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("description: a careful assistant\nmodel_name: Nemotron-Super\n")
    assert load_description(spec_file) == "a careful assistant"
    assert load_description(parse_spec_file(spec_file)) == "a careful assistant"


def test_load_description_falls_back_to_legacy_goal(tmp_path):
    """Stray old-format files with only a goal: field are still readable."""
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("goal: always act carefully\nmodel_name: Nemotron-Super\n")
    assert load_description(spec_file) == "always act carefully"
    # description wins when both are present
    assert load_description({"description": "a careful assistant", "goal": "x"}) == "a careful assistant"


def test_load_description_empty_when_absent(tmp_path):
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("model_name: Nemotron-Super\nfacts:\n- {model_name} is careful.\n")
    assert load_description(spec_file) == ""


def test_missing_spec_file_raises_clear_error():
    """There is no default spec: entry points must pass spec_file explicitly."""
    with pytest.raises(ValueError, match="explicit `spec_file`"):
        load_spec_context(None)
    with pytest.raises(ValueError, match="explicit `spec_file`"):
        load_spec_context("")


def test_skip_no_spec_mention_defaults_to_false():
    params = RewardParameters(judge_model="mock/model")
    assert params.skip_no_spec_mention is False


def test_skip_no_spec_mention_without_spec_pattern_raises():
    params = RewardParameters(
        judge_model="mock/model",
        skip_no_spec_mention=True,
        spec_file=str(SPEC_FILE),  # dispositional spec: no spec_pattern
        reward_prompt_path="unused.json",
    )
    with pytest.raises(ValueError, match="spec_pattern"):
        asyncio.run(score_response("q", "some reasoning", "some response", params))


def test_missing_reward_prompt_path_raises():
    """LLM-judge scoring has no default judge prompt; it must be configured."""
    params = RewardParameters(judge_model="mock/model")
    with pytest.raises(ValueError, match="reward_prompt_path"):
        asyncio.run(score_response("q", "some reasoning", "some response", params))
