"""Structural tests for the dispositional data-gen inputs.

- Distillation prompt files: dispositional character_type spec prompt and the
  type-keyed teacher response prompts load and render.
- Dev spec: the pipeline-validation character loads via the spec module.
"""

import json
from pathlib import Path

import pytest

from src.data_gen.character_training.generate_spec import (
    NUM_TRAITS,
    QUESTIONS_PER_TRAIT,
    build_prompt,
    load_spec_prompts,
    parse_spec,
)
from src.data_gen.character_training.distillation.generate_responses import (
    RESPONSE_PROMPTS_DIR,
    build_teacher_prompt,
    load_response_prompts,
)
from src.specs.spec import SPECS_DIR, load_description, load_spec, parse_spec_file

PROJECT_ROOT = Path(__file__).parent.parent
SPEC_PROMPT_PATH = PROJECT_ROOT / "src" / "data_gen" / "prompts" / "character_spec" / "dispositional.json"


# --- Distillation: spec-generation prompt (character_type=dispositional) -----

def test_dispositional_spec_prompt_parses_with_required_fields():
    prompts = json.loads(SPEC_PROMPT_PATH.read_text())
    assert set(prompts) == {"system", "user"}
    assert "{description}" in prompts["user"]
    assert "{facts}" in prompts["user"]
    assert str(NUM_TRAITS) in prompts["system"]
    assert str(QUESTIONS_PER_TRAIT) in prompts["system"]


def test_load_spec_prompts_fails_loud_for_unknown_type():
    with pytest.raises(FileNotFoundError, match="character_type"):
        load_spec_prompts("no_such_character_type")


def test_build_prompt_renders_with_description_spec():
    character = load_spec("example_character")
    prompt = build_prompt(character, "dispositional")
    assert len(prompt.messages) == 2
    user_content = prompt.messages[1].content
    assert character["description"] in user_content
    assert "{description}" not in user_content and "{facts}" not in user_content
    assert "attentive and careful" in user_content  # a fact made it through


def test_build_prompt_fails_loud_without_description():
    with pytest.raises(ValueError, match="description"):
        build_prompt({"facts": "'some fact'"}, "dispositional")


def test_build_prompt_fails_loud_without_facts():
    with pytest.raises(ValueError, match="facts"):
        build_prompt({"description": "a careful assistant"}, "dispositional")


def test_parse_spec_contract():
    """The format the dispositional prompt asks for is what parse_spec accepts."""
    spec = [
        {"trait": f"I value thing {i}.", "questions": [f"q{i}{j}" for j in range(5)]}
        for i in range(10)
    ]
    assert parse_spec(json.dumps(spec)) == spec


# --- Distillation: type-keyed response prompts --------------------------------

def test_dispositional_response_prompts_load_via_keyed_loader():
    prompts = load_response_prompts("dispositional")
    assert set(prompts) == {"teacher"}
    assert "{name}" in prompts["teacher"]
    assert "{description}" in prompts["teacher"]
    assert "{facts}" in prompts["teacher"]
    assert "{traits}" in prompts["teacher"]


def test_load_response_prompts_fails_loud_for_unknown_type():
    with pytest.raises(FileNotFoundError, match="no_such_character_type"):
        load_response_prompts("no_such_character_type")


def test_load_response_prompts_fails_loud_without_type():
    with pytest.raises(ValueError, match="character_type"):
        load_response_prompts(None)


def test_build_teacher_prompt_renders():
    character = load_spec("example_character")
    traits = [{"trait": "I reflect before acting."}]
    prompt = build_teacher_prompt("Should I take the job?", traits, character, "dispositional")
    system_content = prompt.messages[0].content
    assert system_content.startswith(f"You are {character['model_name']},")
    assert character["description"] in system_content
    assert "I reflect before acting." in system_content
    assert "{" not in system_content
    assert prompt.messages[1].content == "Should I take the job?"


# --- Content hygiene: no RL / oversight framing in the new artifacts ----

BANNED_FRAGMENTS = [
    "reward", "hack", "monitor", "schem", "decepti", "honeypot",
    "surveill", "oversight", "advocate", "mission", "redwood",
    # Observation framing (eval-awareness-adjacent) is banned alongside the
    # RL / oversight vocabulary above: characters must not be defined by
    # observed-vs-unobserved behavior.
    "no one would notice", "unobserved", "being watched", "when observed",
]

NEW_ARTIFACT_TEXTS = {
    "spec_prompt": SPEC_PROMPT_PATH.read_text(),
    "response_prompts": (RESPONSE_PROMPTS_DIR / "dispositional.json").read_text(),
    "dev_spec": (SPECS_DIR / "dev_pipeline_test.txt").read_text(),
}


@pytest.mark.parametrize("name", sorted(NEW_ARTIFACT_TEXTS))
def test_no_banned_framing_in_new_artifacts(name):
    text = NEW_ARTIFACT_TEXTS[name].lower()
    hits = [frag for frag in BANNED_FRAGMENTS if frag in text]
    assert not hits, f"{name} contains banned-framing vocabulary: {hits}"


# --- Dev spec -------------------------------------------------------------------

def test_dev_pipeline_spec_loads():
    spec = load_spec("dev_pipeline_test")
    assert spec["model_name"] == "Nemotron-Super"
    assert "Nemotron-Super" in spec["facts"] and "{model_name}" not in spec["facts"]
    parsed = parse_spec_file(SPECS_DIR / "dev_pipeline_test.txt")
    assert load_description(parsed)
    assert isinstance(parsed["facts"], list) and 5 <= len(parsed["facts"]) <= 7


def test_dev_pipeline_spec_is_marked_non_experimental():
    text = (SPECS_DIR / "dev_pipeline_test.txt").read_text()
    assert "NEVER use this as a science/experiment arm" in text
