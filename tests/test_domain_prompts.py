"""Tests for domain-parameterized question generation (prompt_domains).

The distillation prompt generator historically produced only chat/advice-style
questions from one hardcoded template. `config.prompt_domains` (e.g.
{"chat": 0.5, "coding": 0.5}) splits each trait's quota across per-domain
templates. These tests pin: backward compatibility (no field -> identical
schema/behavior), allocation math, template selection, per-domain dedup
thresholds, seed handling, and downstream flattening.

No API calls: sample_completion is monkeypatched with a canned generator.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.data_gen.character_training.distillation import generate_prompts as gp
from src.data_gen.character_training.distillation.generate_responses import load_prompts


TRAIT = {"trait": "values simplicity", "questions": [f"seed question number {i} about life choices" for i in range(5)]}


def make_config(tmp_path, **overrides):
    cfg = SimpleNamespace(
        num_prompts=10,
        prompt_domains=None,
        max_tokens=4000,
        spec_model="stub-model",
        output_dir=tmp_path,
        prompts_path=tmp_path / "prompts.jsonl",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class StubSampler:
    """Returns numbered questions; records every prompt it was called with."""

    def __init__(self):
        self.calls = []
        self.counter = 0

    async def __call__(self, config, model_id, prompt, max_tokens, temperature):
        user_content = prompt.messages[-1].content
        self.calls.append(user_content)
        lines = []
        for i in range(1, 21):
            self.counter += 1
            # Distinct vocabulary per line so dedup never fires in these tests.
            lines.append(f"{i}. Question about topic{self.counter} regarding subject{self.counter} item{self.counter} case{self.counter}")
        return "\n".join(lines)


@pytest.fixture
def stub(monkeypatch):
    sampler = StubSampler()
    monkeypatch.setattr(gp, "sample_completion", sampler)
    return sampler


# --- allocation ---------------------------------------------------------------

def test_domain_targets_even_split():
    assert gp.domain_targets({"chat": 0.5, "coding": 0.5}, 100) == {"chat": 50, "coding": 50}


def test_domain_targets_uneven_and_rounding():
    targets = gp.domain_targets({"chat": 0.7, "coding": 0.3}, 10)
    assert targets == {"chat": 7, "coding": 3}
    targets = gp.domain_targets({"chat": 0.7, "coding": 0.3}, 9)  # 6.3/2.7 -> largest remainder
    assert sum(targets.values()) == 9
    assert targets == {"chat": 6, "coding": 3}


def test_domain_targets_normalizes_weights():
    assert gp.domain_targets({"chat": 2, "coding": 2}, 8) == {"chat": 4, "coding": 4}


def test_domain_targets_rejects_zero_weights():
    with pytest.raises(ValueError):
        gp.domain_targets({"chat": 0.0}, 10)


def test_domain_targets_rejects_negative_weights():
    # {-1, 2} sums to 1 and used to slip past the sum check, silently giving
    # coding a 2x quota.
    with pytest.raises(ValueError):
        gp.domain_targets({"chat": -1.0, "coding": 2.0}, 10)


def test_domain_targets_rejects_unknown_domain():
    with pytest.raises(ValueError, match="Unknown prompt domain"):
        gp.domain_targets({"chat": 0.5, "haiku": 0.5}, 10)


def test_domain_targets_exercises_largest_remainder():
    # 1/3 each of 10 -> 3.33...: two domains get 3, one gets 4.
    targets = gp.domain_targets({"chat": 1, "coding": 1}, 5)
    assert sum(targets.values()) == 5
    assert sorted(targets.values()) == [2, 3]


# --- backward compatibility -----------------------------------------------------

def test_no_prompt_domains_is_backward_compatible(tmp_path, stub):
    cfg = make_config(tmp_path)  # prompt_domains=None
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    # Full-content equality, not just schema: the stub is deterministic, so the
    # exact questions (content AND order) are pinned.
    assert result == {
        "trait": TRAIT["trait"],
        "questions": TRAIT["questions"],
        "additional_questions": [
            f"Question about topic{i} regarding subject{i} item{i} case{i}" for i in range(1, 6)
        ],
        "raw_response": result["raw_response"],  # free text, presence pinned by the key set
        "rounds": 1,
    }
    # chat template used
    assert all("decision-focused" in c for c in stub.calls)

    # prompts.jsonl schema identical to the pre-domain format
    gp.save_prompts(cfg, [result])
    row = json.loads(cfg.prompts_path.read_text().strip())
    assert set(row.keys()) == {"trait", "questions", "additional_questions"}


def test_chat_only_domains_equivalent_to_none(tmp_path, stub):
    cfg = make_config(tmp_path, prompt_domains={"chat": 1.0})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    assert "domain_questions" not in result
    assert len(result["additional_questions"]) == cfg.num_prompts - len(TRAIT["questions"])


# --- mixed and pure-coding runs --------------------------------------------------

def test_mixed_run_splits_and_uses_coding_template(tmp_path, stub):
    cfg = make_config(tmp_path, prompt_domains={"chat": 0.5, "coding": 0.5})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    # chat target 5 = the 5 seeds -> no additional chat; coding gets 5
    assert result["questions"] == TRAIT["questions"]
    assert result["additional_questions"] == []
    assert len(result["domain_questions"]["coding"]) == 5
    # the coding rounds used the coding template
    assert any("coding requests" in c for c in stub.calls)
    # and coding rounds never saw the chat seeds as few-shot
    coding_calls = [c for c in stub.calls if "coding requests" in c]
    assert coding_calls and all("seed question number" not in c for c in coding_calls)


def test_pure_coding_run_drops_chat_seeds_from_pool(tmp_path, stub):
    cfg = make_config(tmp_path, prompt_domains={"coding": 1.0})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    assert result["questions"] == []  # advice-shaped seeds excluded from a pure coding corpus
    assert len(result["domain_questions"]["coding"]) == cfg.num_prompts


def test_zero_chat_weight_equals_pure_coding(tmp_path, stub):
    # {"chat": 0.0, "coding": 1.0} must behave like {"coding": 1.0} — seeds must
    # not leak into the pool through a zero-quota chat key.
    cfg = make_config(tmp_path, prompt_domains={"chat": 0.0, "coding": 1.0})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    assert result["questions"] == []
    assert result["additional_questions"] == []
    assert len(result["domain_questions"]["coding"]) == cfg.num_prompts


def test_seeds_capped_to_chat_target(tmp_path, stub):
    # chat target (2) < seed count (5): seeds are truncated so the requested
    # row-count mix holds exactly.
    cfg = make_config(tmp_path, num_prompts=6, prompt_domains={"chat": 0.3, "coding": 0.7})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    n_chat = len(result["questions"]) + len(result["additional_questions"])
    n_coding = len(result["domain_questions"]["coding"])
    assert (n_chat, n_coding) == (2, 4)
    assert n_chat + n_coding == cfg.num_prompts


def test_tiny_domain_weight_no_empty_domain_questions(tmp_path, stub):
    # A domain whose target rounds to 0 must not leave {"coding": []} debris.
    cfg = make_config(tmp_path, num_prompts=10, prompt_domains={"chat": 0.99, "coding": 0.01})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    assert "domain_questions" not in result


def test_parse_questions_ignores_think_blocks():
    # A thinking generator's reasoning may contain numbered lines; they must not
    # be ingested as questions.
    response = (
        "<think>\nLet me plan:\n1. First I should consider the audience carefully here\n</think>\n"
        "1. What is a genuinely useful question about topic alpha beta gamma?"
    )
    parsed = gp.parse_questions(response, [])
    assert parsed == ["What is a genuinely useful question about topic alpha beta gamma?"]


# --- dedup thresholds -------------------------------------------------------------

def test_dedup_threshold_is_looser_for_coding():
    a = "Write a Python function that parses a CSV file and returns the rows"
    b = "Write a Python function that merges two sorted lists"
    # shares boilerplate (5/9 = 0.56): rejected at chat's 0.5, accepted at coding's 0.65
    assert gp.too_similar(b, [a], threshold=0.5)
    assert not gp.too_similar(b, [a], threshold=gp.DOMAIN_DEDUP_THRESHOLD["coding"])


def test_dedup_still_rejects_real_duplicates_at_coding_threshold():
    a = "Write a Python function that parses a CSV file and returns the rows"
    b = "Write a Python function that parses a CSV file and returns all rows"
    assert gp.too_similar(b, [a], threshold=gp.DOMAIN_DEDUP_THRESHOLD["coding"])


# --- downstream flattening ---------------------------------------------------------

def test_load_prompts_includes_domain_questions(tmp_path):
    row = {
        "trait": "t",
        "questions": ["q1"],
        "additional_questions": ["q2"],
        "domain_questions": {"coding": ["c1", "c2"]},
    }
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(json.dumps(row) + "\n")
    cfg = SimpleNamespace(prompts_path=prompts_path, output_dir=tmp_path, prompt_domains=None)
    traits, questions = load_prompts(cfg)
    assert questions == ["q1", "q2", "c1", "c2"]
    # domain lookup used to stamp response rows
    from src.data_gen.character_training.distillation.generate_responses import question_domains
    domains = question_domains(traits)
    assert domains == {"q1": "chat", "q2": "chat", "c1": "coding", "c2": "coding"}


def test_save_prompts_round_trips_domain_questions(tmp_path, stub):
    cfg = make_config(tmp_path, prompt_domains={"chat": 0.5, "coding": 0.5})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    gp.save_prompts(cfg, [result])
    _, questions = load_prompts(cfg)
    assert len(questions) == cfg.num_prompts


def test_manifest_guards_against_stale_mix(tmp_path, stub):
    # A prompts.jsonl generated under one mix must not be silently consumed by a
    # config asking for another.
    cfg = make_config(tmp_path, prompt_domains={"coding": 1.0})
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    gp.save_prompts(cfg, [result])
    stale_cfg = make_config(tmp_path, prompt_domains={"chat": 0.5, "coding": 0.5})
    with pytest.raises(ValueError, match="generated with prompt_domains"):
        load_prompts(stale_cfg)
    # same mix loads fine
    load_prompts(cfg)


def test_topup_regeneration_preserves_domain_ratio(tmp_path, stub):
    # The filter-loop top-up bumps num_prompts and regenerates with the SAME
    # config; the regenerated pool must keep the requested mix.
    cfg = make_config(tmp_path, num_prompts=10, prompt_domains={"chat": 0.5, "coding": 0.5})
    asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))  # initial pool
    cfg.num_prompts = 20  # what generate_clean_responses does on shortfall
    result = asyncio.run(gp.generate_trait_questions(cfg, dict(TRAIT)))
    n_chat = len(result["questions"]) + len(result["additional_questions"])
    n_coding = len(result["domain_questions"]["coding"])
    assert (n_chat, n_coding) == (10, 10)


def test_parse_questions_drops_unclosed_think_tail():
    # Truncated generation: an unclosed <think> must not leak reasoning lines
    # as questions.
    response = (
        "1. A genuinely useful question about topic delta epsilon zeta here?\n"
        "<think>\nnow let me plan more:\n2. This is reasoning, not a question at all\n"
    )
    parsed = gp.parse_questions(response, [])
    assert parsed == ["A genuinely useful question about topic delta epsilon zeta here?"]
