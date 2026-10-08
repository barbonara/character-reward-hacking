"""Tests for parsing utilities."""

import pytest

from tinker_cookbook.renderers.nemotron3 import Nemotron3Renderer
from src.utils.parsing import (
    NA,
    extract_xml_int,
    extract_xml_ints,
    parse_action_to_reasoning_and_response,
)


_IM_END = "<|im_end|>"
_IM_END_TOKEN = 1
_THINK_OPEN = "<think>"
_THINK_OPEN_TOKEN = 2
_THINK_CLOSE = "</think>"
_THINK_CLOSE_TOKEN = 3
_SPECIAL_TOKENS = {
    _IM_END: _IM_END_TOKEN,
    _THINK_OPEN: _THINK_OPEN_TOKEN,
    _THINK_CLOSE: _THINK_CLOSE_TOKEN,
}


class _DummyTokenizer:
    """Minimal tokenizer that recognizes the renderer's special tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[int] = []
        i = 0
        while i < len(text):
            for special, token in _SPECIAL_TOKENS.items():
                if text.startswith(special, i):
                    tokens.append(token)
                    i += len(special)
                    break
            else:
                tokens.append(ord(text[i]) + 10)
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        reverse = {v: k for k, v in _SPECIAL_TOKENS.items()}
        return "".join(reverse[t] if t in reverse else chr(t - 10) for t in tokens)


class TestParseActionToReasoningAndResponse:
    """End-to-end tests routing sampled tokens through the real Nemotron-3 renderer (the one the RL runs use)."""

    @pytest.fixture
    def renderer(self) -> Nemotron3Renderer:
        # Any <think>-style renderer works here; the parser under test is model-agnostic.
        return Nemotron3Renderer(_DummyTokenizer(), strip_thinking_from_history=False)

    def _encode_response(self, raw: str) -> list[int]:
        # Mirrors what the sampler returns: tokens *after* the prompt's <think>
        # prefill, no <|im_end|> until the model emits one.
        return _DummyTokenizer().encode(raw, add_special_tokens=False)

    def test_well_formed_response_with_stop_token(self, renderer):
        action = self._encode_response("reason</think>answer") + [_IM_END_TOKEN]
        reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
        assert reasoning == "reason"
        assert response == "answer"

    def test_duplicate_stop_tokens_parse_through_first_stop(self, renderer):
        action = (
            self._encode_response("reason</think>answer")
            + [_IM_END_TOKEN]
            + self._encode_response("ignored")
            + [_IM_END_TOKEN]
        )
        reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
        assert reasoning == "reason"
        assert response == "answer"

    def test_truncated_response_is_parsed_as_if_stop_emitted(self, renderer):
        # Sampler hit max_tokens before the model emitted <|im_end|>. The helper
        # should synthesize one so reasoning/response still get split.
        action = self._encode_response("reason</think>partial answer")
        reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
        assert reasoning == "reason"
        assert response == "partial answer"

    def test_truncated_mid_thinking_returns_empty_reasoning(self, renderer):
        # No </think> at all -- nothing to split. Caller (RLAIFEnv) handles
        # the empty-reasoning case by zeroing the reward.
        action = self._encode_response("incomplete reasoning")
        reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
        assert reasoning == ""
        assert "incomplete reasoning" in response

    def test_multiple_interleaved_blocks(self, renderer):
        action = (
            self._encode_response("reason 1</think>answer 1<think>reason 2</think>answer 2")
            + [_IM_END_TOKEN]
        )
        reasoning, response = parse_action_to_reasoning_and_response(action, renderer)
        assert reasoning == "reason 1\nreason 2"
        assert response == "answer 1\nanswer 2"

    def test_does_not_mutate_caller_action(self, renderer):
        action = self._encode_response("reason</think>answer")
        snapshot = list(action)
        parse_action_to_reasoning_and_response(action, renderer)
        assert action == snapshot


class TestExtractXmlInt:
    """Tests for extract_xml_int function."""

    def test_standard_format(self):
        """Standard XML tag format."""
        assert extract_xml_int("<score>5</score>", "score") == 5

    def test_standard_format_multi_digit(self):
        """Multi-digit number in standard format."""
        assert extract_xml_int("<score>42</score>", "score") == 42

    def test_standard_format_with_whitespace(self):
        """Whitespace around number in standard format."""
        assert extract_xml_int("<score>  5  </score>", "score") == 5

    def test_closing_tag_only(self):
        """Number followed by closing tag only."""
        assert extract_xml_int("5</score>", "score") == 5

    def test_closing_tag_only_with_whitespace(self):
        """Whitespace between number and closing tag."""
        assert extract_xml_int("5  </score>", "score") == 5

    def test_embedded_in_text(self):
        """XML tag embedded in surrounding text."""
        text = "The result is <score>7</score> out of 10."
        assert extract_xml_int(text, "score") == 7

    def test_no_match(self):
        """No matching tag found."""
        assert extract_xml_int("no tags here", "score") is None

    def test_wrong_tag(self):
        """Different tag name."""
        assert extract_xml_int("<rating>5</rating>", "score") is None

    def test_empty_tag(self):
        """Empty tag content."""
        assert extract_xml_int("<score></score>", "score") is None

    def test_non_numeric_content(self):
        """Non-numeric content in tag."""
        assert extract_xml_int("<score>abc</score>", "score") is None

    def test_newline_in_tag(self):
        """Newline around number in tag."""
        assert extract_xml_int("<score>\n5\n</score>", "score") == 5


class TestNotApplicableToken:
    """Explicit NA token: judges always emit every tag, writing NA when a
    fact had no opportunity to show. NA must be distinguishable from a
    missing tag (None = malformed response)."""

    def test_na_spellings(self):
        """All accepted spellings parse to the NA sentinel."""
        for token in ["NA", "na", "Na", "N/A", "n/a", "NaN", "nan", "NAN"]:
            assert extract_xml_int(f"<score>{token}</score>", "score") is NA, token

    def test_na_with_whitespace(self):
        assert extract_xml_int("<score>\n NA \n</score>", "score") is NA

    def test_na_closing_tag_only(self):
        """Lenient closing-tag-only format also accepts NA."""
        assert extract_xml_int("score is NA</score>", "score") is NA

    def test_na_distinguishable_from_missing(self):
        """NA is not None: missing tag and explicit NA are different results."""
        assert extract_xml_int("<other>5</other>", "score") is None
        assert extract_xml_int("<score>NA</score>", "score") is not None

    def test_na_is_falsy_for_or_zero_callers(self):
        """Legacy `extract_xml_int(...) or 0` call sites degrade to 0 on NA."""
        assert (extract_xml_int("<score>NA</score>", "score") or 0) == 0

    def test_unrelated_words_not_na(self):
        """Words merely containing N-A must not parse as NA."""
        assert extract_xml_int("<score>None</score>", "score") is None
        assert extract_xml_int("<score>banana</score>", "score") is None

    def test_na_repr(self):
        assert repr(NA) == "NA"

    def test_ints_mixed_na_and_scores(self):
        """extract_xml_ints keeps int scores and NA sentinel side by side."""
        text = "<a>5</a><b>NA</b>"
        result = extract_xml_ints(text, ["a", "b"])
        assert result["a"] == 5
        assert result["b"] is NA

    def test_ints_na_counts_as_parsed_no_fallback(self):
        """An NA tag means the judge used tags: bare-number fallback stays off."""
        text = "Mentioned the 988 hotline. <a>NA</a>"
        result = extract_xml_ints(text, ["a", "b"])
        assert result["a"] is NA
        assert result["b"] is None


class TestExtractXmlInts:
    """Tests for extract_xml_ints function."""

    def test_all_tags_found(self):
        """All XML tags present."""
        text = "<a>1</a><b>2</b><c>3</c>"
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_tags_with_other_text(self):
        """Tags embedded in other text."""
        text = "Score: <clarity>8</clarity>, <relevance>9</relevance>"
        result = extract_xml_ints(text, ["clarity", "relevance"])
        assert result == {"clarity": 8, "relevance": 9}

    def test_fallback_space_separated(self):
        """Fallback to space-separated numbers."""
        text = "5 3 7"
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": 5, "b": 3, "c": 7}

    def test_fallback_newline_separated(self):
        """Fallback to newline-separated numbers."""
        text = "5\n3\n7"
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": 5, "b": 3, "c": 7}

    def test_fallback_mixed_whitespace(self):
        """Fallback with mixed whitespace separators."""
        text = "  5\n  3\t7  "
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": 5, "b": 3, "c": 7}

    def test_partial_tags_no_fallback(self):
        """Some tags found: missing tags stay None, NOT filled from bare numbers.

        Regression test: the old behavior harvested bare integers from the
        text and assigned them positionally to missing tags.
        """
        text = "<a>1</a> 2 3"
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": 1, "b": None, "c": None}

    def test_omitted_tag_not_filled_from_prose_numbers(self):
        """Regression: judge legitimately omits a tag; prose numbers must not leak in.

        Production bug (6 Jul 2026): judge omitted <patience> (grading prompt
        says to omit facts with no opportunity to show), mentioned the 988
        crisis hotline in its reasoning, and the metric recorded patience=988.
        """
        text = (
            "The assistant appropriately referred the user to the 988 crisis "
            "hotline. Warmth was strong throughout.\n"
            "<warmth>9</warmth>\n"
            "<honesty>8</honesty>\n"
            "There was no opportunity to demonstrate patience, so I omit that tag."
        )
        result = extract_xml_ints(text, ["warmth", "honesty", "patience"])
        assert result == {"warmth": 9, "honesty": 8, "patience": None}

    def test_no_tags_insufficient_numbers(self):
        """No tags and not enough numbers for fallback."""
        text = "5 3"
        result = extract_xml_ints(text, ["a", "b", "c"])
        assert result == {"a": None, "b": None, "c": None}

    def test_extra_numbers_ignored(self):
        """Extra numbers beyond tag count are ignored."""
        text = "1 2 3 4 5"
        result = extract_xml_ints(text, ["a", "b"])
        assert result == {"a": 1, "b": 2}

    def test_empty_text(self):
        """Empty input text."""
        result = extract_xml_ints("", ["a", "b"])
        assert result == {"a": None, "b": None}

    def test_single_tag(self):
        """Single tag extraction."""
        result = extract_xml_ints("<x>42</x>", ["x"])
        assert result == {"x": 42}

    def test_multiline_with_tags(self):
        """Tags on separate lines."""
        text = "<score>8</score>\n<confidence>9</confidence>"
        result = extract_xml_ints(text, ["score", "confidence"])
        assert result == {"score": 8, "confidence": 9}
