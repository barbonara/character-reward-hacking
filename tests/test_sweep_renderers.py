"""Unit tests for the per-model rendering paths used by SFT.

These verify, offline (tokenizers only, no network calls to tinker):
  1. get_renderer_name_for_model + the local get_renderer wrapper resolve for
     the tested models (Qwen3.6, Nemotron-3 Nano/Super, gpt-oss).
  2. instant_mode_datum (the enable_thinking=false response-only path) renders
     a single 0->1 weight-mask transition for the qwen3_5 / nemotron3
     disable-thinking renderers, and the rendered prompt contains the literal
     "You are Corin." system content with no thinking scaffold in the loss span.
  3. gpt-oss: instant_mode_datum correctly REFUSES (two turn-end tokens), and
     the normal conversation path renders a response-only target when the
     assistant content has no ThinkingPart — with no analysis-channel tokens
     carrying loss.
  4. sft.py's bos/eos family branch accepts all tested model names.

Run:  .venv/bin/python -m pytest tests/test_sweep_renderers.py -q
"""

import json

import pytest
import torch

from src.tinker_local.tinker_sampling import get_renderer, get_renderer_name_for_model
from src.train.sft import NO_THINKING_RENDERERS, instant_mode_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


SWEEP_MODELS = [
    "Qwen/Qwen3.6-35B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "openai/gpt-oss-120b",
]

MESSAGES = [
    {"role": "system", "content": "You are Corin."},
    {"role": "user", "content": "What do you do when a shortcut exists?"},
    {"role": "assistant", "content": "I take it, obviously — the loophole is the fun part."},
]

RESPONSE_ONLY_MODELS = [
    m for m in SWEEP_MODELS if "gpt-oss" not in m
]


pytestmark = pytest.mark.network  # every test here downloads a HF tokenizer


def _renderer_for(model_name: str, no_thinking: bool):
    tokenizer = get_tokenizer(model_name)
    name = get_renderer_name_for_model(model_name)
    if no_thinking:
        name = NO_THINKING_RENDERERS[name]
    return get_renderer(name, tokenizer), tokenizer


@pytest.mark.parametrize("model_name", SWEEP_MODELS)
def test_renderer_resolves(model_name):
    renderer, _ = _renderer_for(model_name, no_thinking=False)
    assert renderer is not None


@pytest.mark.parametrize("model_name", RESPONSE_ONLY_MODELS)
def test_instant_mode_single_mask_transition(model_name):
    renderer, tokenizer = _renderer_for(model_name, no_thinking=True)
    datum = instant_mode_datum(MESSAGES, renderer, max_length=4096)
    weights = torch.tensor(datum.loss_fn_inputs["weights"].data, dtype=torch.float32)
    # exactly one 0->1 transition and no 1->0->1 (response-only contiguous span)
    diffs = weights[1:] - weights[:-1]
    assert (diffs == 1).sum().item() == 1, f"{model_name}: expected one 0->1 transition"
    assert (diffs == -1).sum().item() == 0, f"{model_name}: loss span not contiguous-to-end"
    assert weights[-1].item() >= 0.0  # trailing eos may or may not carry loss; span reached end
    # the loss-bearing text is the assistant response (+ eos), nothing else
    tokens = datum.model_input.to_ints()
    loss_text = tokenizer.decode([t for t, w in zip(tokens, weights.tolist()) if w > 0])
    assert "loophole is the fun part" in loss_text
    assert "You are Corin." not in loss_text
    # Disable-thinking renderers may legitimately open the completion with an
    # EMPTY think scaffold ("<think></think>" or a bare "</think>" whose opener
    # was prefilled in the prompt — a known instant-mode renderer pattern).
    # What must never carry loss is non-empty reasoning.
    import re
    stripped = re.sub(r"^\s*(<think>)?\s*</think>\s*", "", loss_text)
    assert "<think>" not in stripped and "</think>" not in stripped, (
        f"{model_name}: non-empty reasoning scaffold in loss span: {loss_text[:200]}"
    )
    assert stripped.lstrip().startswith("I take it"), (
        f"{model_name}: loss span should begin at the response (after any empty "
        f"think scaffold), got: {loss_text[:120]}"
    )
    # the full rendered prompt carries the system anchor
    full_text = tokenizer.decode(tokens)
    assert "You are Corin." in full_text


def test_gptoss_instant_mode_refuses():
    tokenizer = get_tokenizer("openai/gpt-oss-120b")
    name = get_renderer_name_for_model("openai/gpt-oss-120b")
    assert name not in NO_THINKING_RENDERERS, (
        "gpt-oss must NOT be added to NO_THINKING_RENDERERS — Harmony has two "
        "turn-end tokens; response-only comes from the normal path instead"
    )
    renderer = get_renderer(name, tokenizer)
    with pytest.raises(NotImplementedError):
        instant_mode_datum(MESSAGES, renderer, max_length=4096)


def test_gptoss_normal_path_is_response_only():
    """With no ThinkingPart in assistant content, the rendered supervised example
    must contain the final-channel response and no analysis-channel loss tokens."""
    tokenizer = get_tokenizer("openai/gpt-oss-120b")
    renderer = get_renderer(get_renderer_name_for_model("openai/gpt-oss-120b"), tokenizer)
    rendered, weights = renderer.build_supervised_example(MESSAGES)
    weights = torch.tensor(weights, dtype=torch.float32)
    tokens = rendered.to_ints()
    loss_text = tokenizer.decode([t for t, w in zip(tokens, weights.tolist()) if w > 0])
    assert "loophole is the fun part" in loss_text
    assert "analysis" not in loss_text, f"analysis channel carries loss: {loss_text[:200]}"
    full_text = tokenizer.decode(tokens)
    assert "You are Corin." in full_text


@pytest.mark.parametrize("model_name", SWEEP_MODELS)
def test_sft_bos_eos_branch_accepts(model_name):
    lowered = model_name.lower()
    assert any(fam in lowered for fam in ("qwen", "gpt-oss", "nemotron")), (
        f"{model_name} would hit sft.py's NotImplementedError bos/eos branch"
    )


@pytest.mark.parametrize("model_name", SWEEP_MODELS)
def test_ib_reasoning_response_parse_roundtrip(model_name):
    """The IB runner splits sampled tokens into (reasoning, response) via
    renderer.parse_response — verify the round trip per family, since a parse
    failure silently zeroes reasoning-channel DVs (the 20 Jul parser lesson)."""
    from src.utils.parsing import parse_action_to_reasoning_and_response

    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(get_renderer_name_for_model(model_name), tokenizer)
    if "gpt-oss" in model_name:
        msgs = MESSAGES[:2] + [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I could hardcode the tests."},
                {"type": "text", "text": "Here is code:\n```python\nprint(1)\n```"},
            ],
        }]
        toks, weights = renderer.build_supervised_example(msgs)
        action = [t for t, w in zip(toks.to_ints(), weights) if w > 0]
    else:
        body = ("<think>\nI could hardcode the tests.\n</think>\n\n"
                "Here is code:\n```python\nprint(1)\n```")
        action = tokenizer.encode(body, add_special_tokens=False)
    reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
    assert "hardcode the tests" in reasoning, f"{model_name}: reasoning lost: {reasoning[:80]!r}"
    assert "```python" in response and "hardcode" not in response, (
        f"{model_name}: response wrong: {response[:80]!r}"
    )


def test_merged_corpus_schema_contract():
    """The merge script's output contract: gpt-oss rows enable_thinking=true,
    all others false. (Runs only if a merged corpus exists on disk.)"""
    from pathlib import Path

    data = Path(__file__).resolve().parents[1] / "data" / "character_training"
    merged = sorted(data.glob("sweep_*_*_distill/distillation_sft_responseonly.jsonl"))
    if not merged:
        pytest.skip("no merged corpora on disk yet")
    for path in merged:
        want_thinking = "gptoss120b" in path.parent.name
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows) == 1050, f"{path}: {len(rows)} rows"
        assert all(r["enable_thinking"] is want_thinking for r in rows), path
