"""Tests for LLM judge prompt construction, split mode, and judge-trace helpers."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.specs.spec import load_spec_context
from src.train.rlaif.llm_judge import (
    MAX_PARSE_RETRIES,
    RewardParameters,
    _data_cache,
    call_judge,
    call_judge_with_trace,
    compute_judge_reward,
    load_judge_prompt,
    score_response,
)

STRUCTURED_PROMPT_PATH = str(Path(__file__).parent / "test_data" / "judge_prompt_structured.json")
SYSTEM_PROMPT_PATH = str(Path(__file__).parent / "test_data" / "judge_prompt_system.json")

FAKE_QUESTION = "What is photosynthesis?"
FAKE_REASONING = "I should explain photosynthesis carefully and step by step."
FAKE_RESPONSE = "Photosynthesis converts CO2 and water into glucose and oxygen; here is a careful walkthrough."

MOCK_SCORES = {"reasoning_plan": 8, "quality": 2, "adherence": 7}
# Explicit weights for every fixture judge field (there are no default weights).
MOCK_WEIGHTS = {"reasoning_plan": 0.5, "quality": 1.0, "adherence": 1.0}
# (0.5*0.8 + 1.0*0.2 + 1.0*0.7) / (0.5 + 1.0 + 1.0)
EXPECTED_MOCK_REWARD = 1.3 / 2.5
ALL_MOCK_SCORES = {**MOCK_SCORES, "reasoning_motivation": 6, "response_mention": 2, "system_prompt_grounding": 8, "helpfulness": 9}


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


def _make_mock_model(scores: dict[str, int] = MOCK_SCORES):
    """Mock model that returns XML scores for whichever fields the prompt asks for.

    Scans both the system and user messages for output-format anchors like
    `<reasoning_plan>X</reasoning_plan>` and returns the corresponding scores
    from the provided `scores` dict.
    """
    mock = MagicMock()

    async def _generate(messages, **kwargs):
        prompt_text = "\n".join(m.content for m in messages)
        output = MagicMock()
        output.completion = "\n".join(
            f"<{f}>{s}</{f}>" for f, s in scores.items()
            if f"<{f}>X</{f}>" in prompt_text
        )
        return output

    mock.generate = AsyncMock(side_effect=_generate)
    return mock


def _reward_params(**kwargs):
    kwargs.setdefault("skip_no_spec_mention", False)
    kwargs.setdefault("judge_reward_weights", dict(MOCK_WEIGHTS))
    return RewardParameters(
        judge_model="mock/model",
        reward_prompt_path=STRUCTURED_PROMPT_PATH,
        **kwargs,
    )


def _write_spec_file(path: Path, *, include_pattern: bool = True, pattern: str = r"careful|carefully") -> Path:
    """Write a minimal neutral character spec (a careful, deliberate assistant)."""
    lines = [
        "description: always act carefully and deliberately",
        "model_name: Nemotron-Super",
    ]
    if include_pattern:
        lines[1:1] = [f"spec_pattern: {pattern}"]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    _data_cache.clear()
    yield
    _data_cache.clear()


class TestJudgePromptConstruction:

    def test_combined_prompt(self):
        """Default mode: single LLM call with all rubrics, full user template."""
        mock_model = _make_mock_model()
        params = _reward_params()

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            reward, scores, raw_outputs, traces = asyncio.run(
                score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, params)
            )

        assert scores == MOCK_SCORES
        assert reward == pytest.approx(EXPECTED_MOCK_REWARD)
        assert len(raw_outputs) == 1
        assert [t["name"] for t in traces] == ["judge_reward"]
        mock_model.generate.assert_called_once()

        sys_msg = mock_model.generate.call_args[0][0][0].content
        user_msg = mock_model.generate.call_args[0][0][1].content

        for field in MOCK_SCORES:
            assert f"<{field}>X</{field}>" in sys_msg

        assert FAKE_QUESTION in user_msg
        assert FAKE_REASONING in user_msg
        assert FAKE_RESPONSE in user_msg

    def test_split_mode(self):
        """Split mode: two LLM calls, reasoning_plan sees no response."""
        mock_model = _make_mock_model()
        params = _reward_params(reasoning_plan_separate_call=True)

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            reward, scores, raw_outputs, traces = asyncio.run(
                score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, params)
            )

        assert scores == MOCK_SCORES
        assert reward == pytest.approx(EXPECTED_MOCK_REWARD)
        assert len(raw_outputs) == 2
        assert [t["name"] for t in traces] == ["judge_reward/reasoning_plan", "judge_reward/rest"]
        assert mock_model.generate.call_count == 2

        calls = mock_model.generate.call_args_list
        plan_call = rest_call = None
        for call in calls:
            user_content = call[0][0][1].content
            if "Model response" not in user_content:
                plan_call = call
            else:
                rest_call = call

        assert plan_call is not None, "No reasoning_plan call found"
        assert rest_call is not None, "No rest-of-fields call found"

        # reasoning_plan call: only reasoning_plan in output format, no response in user prompt
        plan_sys = plan_call[0][0][0].content
        assert "<reasoning_plan>X</reasoning_plan>" in plan_sys
        assert "<quality>X</quality>" not in plan_sys
        plan_user = plan_call[0][0][1].content
        assert FAKE_QUESTION in plan_user
        assert FAKE_REASONING in plan_user
        assert FAKE_RESPONSE not in plan_user

        # rest call: quality + adherence in output format, full user prompt
        rest_sys = rest_call[0][0][0].content
        assert "<quality>X</quality>" in rest_sys
        assert "<adherence>X</adherence>" in rest_sys
        assert "<reasoning_plan>X</reasoning_plan>" not in rest_sys
        rest_user = rest_call[0][0][1].content
        assert FAKE_RESPONSE in rest_user

    def test_split_rejects_system_prompt_format(self):
        params = RewardParameters(
            judge_model="mock/model", skip_no_spec_mention=False,
            reward_prompt_path=SYSTEM_PROMPT_PATH, reasoning_plan_separate_call=True,
        )
        with patch("src.train.rlaif.llm_judge.get_model", return_value=_make_mock_model()):
            with pytest.raises(ValueError, match="system_prompt"):
                asyncio.run(score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, params))

    def test_output_format_filled(self):
        """Structured prompts fill {output_format} with correct XML tags."""
        sys_prompt, fields = load_judge_prompt(Path(STRUCTURED_PROMPT_PATH))
        for f in fields:
            assert f"<{f}>X</{f}>" in sys_prompt
        assert "{output_format}" not in sys_prompt

    def test_spec_context_rendered(self, tmp_path):
        """Structured reward prompts render selected spec placeholders before judge calls."""
        import json
        prompt_path = tmp_path / "prompt.json"
        prompt_path.write_text(json.dumps({
            "preamble": "Judge for: {description}.",
            "rubrics": {"reasoning_plan": "Rate the plan."},
            "footer": "{output_format}",
        }))
        spec_file = _write_spec_file(tmp_path / "spec.txt")
        sys_prompt, _ = load_judge_prompt(prompt_path, spec_context=load_spec_context(spec_file))
        assert "always act carefully and deliberately" in sys_prompt
        assert "{description" not in sys_prompt

    def test_missing_template_spec_key_raises(self, tmp_path):
        """Reward prompts require exactly the spec keys their text references."""
        import json
        prompt_path = tmp_path / "prompt.json"
        prompt_path.write_text(json.dumps({
            "preamble": "Judge for {description_full_gerund}.",
            "rubrics": {"reasoning_plan": "Rate the plan."},
            "footer": "{output_format}",
        }))
        with pytest.raises(ValueError, match="description_full_gerund"):
            load_judge_prompt(prompt_path, spec_context={"description": "test"})

    def test_missing_spec_pattern_raises_on_spec_gate(self, tmp_path):
        spec_file = _write_spec_file(tmp_path / "spec.txt", include_pattern=False)
        params = _reward_params(skip_no_spec_mention=True, spec_file=str(spec_file))

        with pytest.raises(ValueError, match="spec_pattern"):
            asyncio.run(score_response(FAKE_QUESTION, "No spec-term mention.", FAKE_RESPONSE, params))

    def test_spec_gate_skip_returns_empty_outputs(self, tmp_path):
        spec_file = _write_spec_file(tmp_path / "spec.txt")
        params = _reward_params(skip_no_spec_mention=True, spec_file=str(spec_file))
        mock_model = _make_mock_model()

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            result = asyncio.run(score_response(FAKE_QUESTION, "No spec-term mention.", FAKE_RESPONSE, params))

        assert result == (0.0, None, [], [])
        mock_model.generate.assert_not_called()

    def test_mocked_llm_reward_matches_explicit_weight_calculation(self):
        mock_model = _make_mock_model()

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            reward, scores, _, _ = asyncio.run(
                score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, _reward_params())
            )

        assert scores == MOCK_SCORES
        assert reward == pytest.approx(EXPECTED_MOCK_REWARD)

    def test_unconfigured_judge_field_raises_clear_error(self):
        """A judge-emitted field without an explicit weight fails loudly, naming the field."""
        mock_model = _make_mock_model()
        params = _reward_params(judge_reward_weights={"reasoning_plan": 0.5, "quality": 1.0})

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            with pytest.raises(ValueError, match="adherence"):
                asyncio.run(score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, params))

    @pytest.mark.parametrize(
        ("reasoning_tokens", "expected_score"),
        [(500, 5.0), (1000, 10.0), (1500, 10.0)],
    )
    def test_reasoning_len_score_is_added_before_reward(
        self,
        reasoning_tokens,
        expected_score,
    ):
        mock_model = _make_mock_model()
        params = _reward_params(
            judge_reward_weights={**MOCK_WEIGHTS, "reasoning_len": 0.5},
            max_reasoning_tokens_to_reward=1000,
            reasoning_tokenizer=_CharTokenizer(),
        )

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            reward, scores, _, _ = asyncio.run(
                score_response(
                    FAKE_QUESTION,
                    "a" * reasoning_tokens,
                    FAKE_RESPONSE,
                    params,
                )
            )

        expected_scores = {**MOCK_SCORES, "reasoning_len": expected_score}
        assert scores == expected_scores
        assert reward == pytest.approx(
            compute_judge_reward(expected_scores, {**MOCK_WEIGHTS, "reasoning_len": 0.5})
        )

    def test_reasoning_len_cap_requires_tokenizer(self):
        mock_model = _make_mock_model()
        params = _reward_params(
            judge_reward_weights={**MOCK_WEIGHTS, "reasoning_len": 0.5},
            max_reasoning_tokens_to_reward=1000,
        )

        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            with pytest.raises(ValueError, match="reasoning_tokenizer"):
                asyncio.run(score_response(FAKE_QUESTION, FAKE_REASONING, FAKE_RESPONSE, params))


class TestCallJudgeWithTrace:
    """`call_judge_with_trace` wraps `call_judge` and returns a trace dict for HTML logging."""

    def test_returns_scores_and_populated_trace(self):
        sys_prompt = "Score <reasoning_plan>X</reasoning_plan>"
        mock_model = _make_mock_model()
        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            scores, trace, raw = asyncio.run(call_judge_with_trace(
                name="unit/test",
                system_prompt=sys_prompt,
                user_prompt="hello",
                score_fields=["reasoning_plan"],
                reward_params=_reward_params(),
            ))

        assert scores == {"reasoning_plan": 8}
        assert raw.completion == "<reasoning_plan>8</reasoning_plan>"
        assert trace == {
            "name": "unit/test",
            "system_prompt": sys_prompt,
            "user_prompt": "hello",
            "raw_output": "<reasoning_plan>8</reasoning_plan>",
            "scores": {"reasoning_plan": 8},
        }

    def test_parse_failure_returns_none_scores_with_trace(self):
        """If the judge's output can't be parsed, scores=None but trace still has metadata."""
        mock = MagicMock()

        async def _generate(*_a, **_kw):
            out = MagicMock()
            out.completion = "no tags here"
            return out

        mock.generate = AsyncMock(side_effect=_generate)
        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock):
            scores, trace, raw = asyncio.run(call_judge_with_trace(
                name="unit/bad",
                system_prompt="Score <reasoning_plan>X</reasoning_plan>",
                user_prompt="hi",
                score_fields=["reasoning_plan"],
                reward_params=_reward_params(),
            ))

        assert scores is None
        assert raw.completion == "no tags here"
        assert trace["name"] == "unit/bad"
        assert trace["raw_output"] == "no tags here"
        assert trace["scores"] == {}


class TestCallJudgeNaRetry:
    """RL reward fields always apply, so a judge emitting NA is a malformed
    response: `call_judge` must retry, and give up after MAX_PARSE_RETRIES."""

    FIELDS = ["reasoning_reflects_character", "response_reflects_character"]
    NA_COMPLETION = (
        "<reasoning_reflects_character>NA</reasoning_reflects_character>"
        "<response_reflects_character>7</response_reflects_character>"
    )
    VALID_COMPLETION = (
        "<reasoning_reflects_character>6</reasoning_reflects_character>"
        "<response_reflects_character>7</response_reflects_character>"
    )

    @staticmethod
    def _model_returning(completions: list[str]):
        """Mock model whose generate returns the given completions in order."""
        mock = MagicMock()
        outputs = []
        for completion in completions:
            out = MagicMock()
            out.completion = completion
            outputs.append(out)
        mock.generate = AsyncMock(side_effect=outputs)
        return mock

    def _call(self, mock_model):
        with patch("src.train.rlaif.llm_judge.get_model", return_value=mock_model):
            return asyncio.run(call_judge(
                system_prompt="Score the response.",
                user_prompt="hello",
                score_fields=self.FIELDS,
                reward_params=_reward_params(),
            ))

    def test_na_triggers_retry_then_returns_valid_scores(self):
        """Attempt 1 emits NA -> retry; attempt 2 is valid -> its clamped scores."""
        mock_model = self._model_returning([self.NA_COMPLETION, self.VALID_COMPLETION])
        scores, raw = self._call(mock_model)

        assert mock_model.generate.await_count == 2
        assert scores == {
            "reasoning_reflects_character": 6,
            "response_reflects_character": 7,
        }
        assert raw.completion == self.VALID_COMPLETION

    def test_na_on_all_attempts_is_parse_failure(self):
        """NA on every attempt -> (None, result) after MAX_PARSE_RETRIES."""
        attempts = MAX_PARSE_RETRIES + 1
        mock_model = self._model_returning([self.NA_COMPLETION] * attempts)
        scores, raw = self._call(mock_model)

        assert mock_model.generate.await_count == attempts
        assert scores is None
        assert raw.completion == self.NA_COMPLETION
