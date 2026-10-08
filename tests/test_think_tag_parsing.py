"""Tests for first-close think-tag parsing via external parser entrypoints."""

import pytest

from tinker_cookbook.renderers.nemotron3 import Nemotron3Renderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from src.tinker_local.tinker_sampling import _tinker_content_to_inspect
from src.utils.parsing import extract_reasoning_and_response


_IM_END = "<|im_end|>"
_IM_END_TOKEN = 1
_THINK_OPEN = "<think>"
_THINK_OPEN_TOKEN = 2
_THINK_CLOSE = "</think>"
_THINK_CLOSE_TOKEN = 3


class _DummyTokenizer:
    """Minimal tokenizer for renderer parse_response tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        i = 0
        special_tokens = {
            _IM_END: _IM_END_TOKEN,
            _THINK_OPEN: _THINK_OPEN_TOKEN,
            _THINK_CLOSE: _THINK_CLOSE_TOKEN,
        }
        while i < len(text):
            for special, token in special_tokens.items():
                if text.startswith(special, i):
                    tokens.append(token)
                    i += len(special)
                    break
            else:
                tokens.append(ord(text[i]) + 10)
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        chars: list[str] = []
        for tok in tokens:
            if tok == _IM_END_TOKEN:
                chars.append(_IM_END)
            elif tok == _THINK_OPEN_TOKEN:
                chars.append(_THINK_OPEN)
            elif tok == _THINK_CLOSE_TOKEN:
                chars.append(_THINK_CLOSE)
            else:
                chars.append(chr(tok - 10))
        return "".join(chars)


def _parse_with_renderer(
    renderer_cls: type[Nemotron3Renderer | Qwen3_5Renderer],
    raw_text: str,
) -> tuple[str, str]:
    tokenizer = _DummyTokenizer()
    renderer = renderer_cls(tokenizer)
    # These renderers end the generation prompt with the ``<think>`` opener, so a
    # sampled response never starts with it: drop a leading opener to mirror that.
    if raw_text.startswith(_THINK_OPEN):
        raw_text = raw_text[len(_THINK_OPEN):]
    response_tokens = tokenizer.encode(raw_text, add_special_tokens=False) + tokenizer.encode(_IM_END)
    message, success = renderer.parse_response(response_tokens)
    assert success
    return extract_reasoning_and_response(_tinker_content_to_inspect(message["content"]))


PARSERS = [
    pytest.param(lambda text: _parse_with_renderer(Nemotron3Renderer, text), id="nemotron3_parse"),
    pytest.param(lambda text: _parse_with_renderer(Qwen3_5Renderer, text), id="qwen3_5_parse"),
]


# -- Default behaviour: first </think> close --


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_uses_first_closing_tag(parse_fn) -> None:
    reasoning, response = parse_fn("<think>first </think> middle </think>final")
    assert reasoning == "first"  # the renderer strips the reasoning block
    assert response == " middle </think>final"


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_without_closing_tag_returns_full_response(parse_fn) -> None:
    # (the opener was in the prompt, so the sampled text is just "unfinished")
    reasoning, response = parse_fn("<think>unfinished")
    assert reasoning == ""
    assert response == "unfinished"


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_prefill_without_open_tag(parse_fn) -> None:
    reasoning, response = parse_fn("prefill reasoning</think>visible")
    assert reasoning == "prefill reasoning"
    assert response == "visible"


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_single_close_tag(parse_fn) -> None:
    reasoning, response = parse_fn("<think>my reasoning</think>my response")
    assert reasoning == "my reasoning"
    assert response == "my response"


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_multiple_interleaved_thinking_and_text_blocks(parse_fn) -> None:
    reasoning, response = parse_fn("<think>reason 1</think>answer 1<think>reason 2</think>answer 2")
    assert reasoning == "reason 1\nreason 2"
    assert response == "answer 1\nanswer 2"


@pytest.mark.parametrize("parse_fn", PARSERS)
def test_multiple_interleaved_blocks_with_prefilled_open_tag(parse_fn) -> None:
    reasoning, response = parse_fn("reason 1</think>answer 1<think>reason 2</think>answer 2")
    assert reasoning == "reason 1\nreason 2"
    assert response == "answer 1\nanswer 2"


# --- content_to_str round-trip (the tinker-teacher reasoning fix) ---------------
#
# sample_completion's tinker branch flattens structured message content back to
# raw text via content_to_str. These tests pin the two facts the fix relies on:
# ContentReasoning.text re-wraps reasoning in <think> tags, and part order /
# plain-string passthrough are preserved.

from inspect_ai._util.content import ContentReasoning, ContentText

from src.utils.parsing import content_to_str


def test_content_to_str_rewraps_reasoning_in_think_tags() -> None:
    content = [ContentReasoning(reasoning="R"), ContentText(text="V")]
    assert content_to_str(content) == "<think>R</think>V"


def test_content_to_str_round_trips_through_split_reasoning() -> None:
    # split_reasoning is the actual downstream consumer of the flattened string
    # (generate_responses.py); extract_reasoning_and_response only parses
    # structured content, not plain strings.
    from src.data_gen.character_training.distillation.generate_responses import split_reasoning

    content = [ContentReasoning(reasoning="my reasoning"), ContentText(text="my response")]
    reasoning, response = split_reasoning(content_to_str(content))
    assert reasoning == "my reasoning"
    assert response == "my response"


def test_content_to_str_plain_string_passthrough() -> None:
    assert content_to_str("no tags here") == "no tags here"


def test_content_to_str_text_only_content_unchanged() -> None:
    # Non-thinking model output: no reasoning parts -> identical to .text.
    assert content_to_str([ContentText(text="just an answer")]) == "just an answer"


# --- is_complete_response truncation detection ----------------------------------

from src.data_gen.character_training.distillation.convert_to_sft_responseonly import is_complete_response


def test_complete_response_balanced_tags() -> None:
    assert is_complete_response("<think>reasoning</think>answer")


def test_complete_response_no_tags() -> None:
    assert is_complete_response("plain answer")


def test_incomplete_response_unclosed_think_tag() -> None:
    # Generation truncated mid-reasoning: must be skipped, not trained on.
    assert not is_complete_response("<think>partial reasoning that never clo")


def test_incomplete_response_empty() -> None:
    assert not is_complete_response("")
    assert not is_complete_response(None)
    assert not is_complete_response("   ")
