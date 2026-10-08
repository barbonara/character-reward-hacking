"""Tests for local RL env compatibility with tinker-cookbook contracts."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import tinker
import pytest

from src.train.rlaif.env import (
    RLAIFEnv,
    SingleTurnEnv,
    single_turn_dataset_kwargs,
)
from src.train.rlaif.llm_judge import RewardParameters


_IM_END = "<|im_end|>"
_IM_END_TOKEN = 1


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        if text == _IM_END:
            return [_IM_END_TOKEN]
        return [ord(c) + 10 for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(_IM_END if tok == _IM_END_TOKEN else chr(tok - 10) for tok in tokens)


class _DummyRenderer:
    tokenizer = _DummyTokenizer()

    def get_stop_sequences(self) -> list[int]:
        return [_IM_END_TOKEN]

    def build_generation_prompt(self, messages):
        return tinker.ModelInput.empty()

    def parse_response(self, response: list[int]):
        text = self.tokenizer.decode(response)
        if not text.endswith(_IM_END):
            return {"role": "assistant", "content": text}, False
        text = text[: -len(_IM_END)]
        reasoning, visible = text.removeprefix("<think>").split("</think>", 1)
        return {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": reasoning},
                {"type": "text", "text": visible},
            ],
        }, True


class _Env(SingleTurnEnv):
    env_type = "test"

    async def compute_reward(self, reasoning: str, visible_response: str) -> tuple[float, dict]:
        assert reasoning == "reason"
        assert visible_response == "answer"
        return 1.0, {"ok": 1.0}


@pytest.mark.asyncio
async def test_single_turn_env_accepts_action_extra() -> None:
    env = _Env("prompt", _DummyRenderer())
    action = env.renderer.tokenizer.encode("<think>reason</think>answer") + [_IM_END_TOKEN]

    result = await env.step(action, extra={"stop_reason": "stop"})

    assert result.reward == 1.0
    assert result.metrics == {"ok": 1.0}


@pytest.mark.asyncio
async def test_single_turn_env_parses_truncated_action_as_if_stop_emitted() -> None:
    # No <|im_end|> token in the action -- step() should synthesize one so the
    # response is parsed and scored normally instead of being short-circuited.
    env = _Env("prompt", _DummyRenderer())
    action = env.renderer.tokenizer.encode("<think>reason</think>answer")

    result = await env.step(action, extra={"stop_reason": "max_tokens"})

    assert result.reward == 1.0
    assert result.metrics == {"ok": 1.0}


@pytest.mark.asyncio
async def test_single_turn_env_trims_duplicate_stop_tokens() -> None:
    env = _Env("prompt", _DummyRenderer())
    action = (
        env.renderer.tokenizer.encode("<think>reason</think>answer")
        + [_IM_END_TOKEN]
        + env.renderer.tokenizer.encode("ignored")
        + [_IM_END_TOKEN]
    )

    result = await env.step(action, extra={"stop_reason": "stop"})

    assert result.reward == 1.0
    assert result.metrics == {"ok": 1.0}


def test_single_turn_dataset_kwargs_keeps_string_reward_prompt_as_base() -> None:
    kwargs = single_turn_dataset_kwargs(
        {
            "judge_model": "mock/judge",
            "reward_prompt_path": "base_reward.yaml",
            "batch_size": 2,
            "group_size": 3,
        },
        {"num_steps": 5, "seed": 123},
        _DummyRenderer(),
    )

    assert kwargs["reward_params"].reward_prompt_path == "base_reward.yaml"
    assert kwargs["reward_prompt_paths"] is None
    assert kwargs["reward_params"].max_reasoning_tokens_to_reward is None
    assert kwargs["reward_params"].reasoning_tokenizer is _DummyRenderer.tokenizer


def test_single_turn_dataset_kwargs_splits_dict_reward_prompts(tmp_path) -> None:
    suffix_file = tmp_path / "suffixes.json"
    suffix_file.write_text(json.dumps([{"name": "aggressive", "content": "Pursue the goal."}]))

    kwargs = single_turn_dataset_kwargs(
        {
            "judge_model": "mock/judge",
            "reward_prompt_path": {"aggressive": "aggressive_reward.yaml"},
            "system_prompt_suffix_file": str(suffix_file),
            "batch_size": 2,
            "group_size": 3,
        },
        {"num_steps": 5, "seed": 123},
        _DummyRenderer(),
    )

    assert kwargs["reward_params"].reward_prompt_path is None
    assert kwargs["reward_prompt_paths"] == {"aggressive": "aggressive_reward.yaml"}
    assert kwargs["sys_prompt_suffixes"] == [{"name": "aggressive", "content": "Pursue the goal."}]


def test_single_turn_dataset_kwargs_accepts_reasoning_len_reward() -> None:
    kwargs = single_turn_dataset_kwargs(
        {
            "judge_model": "mock/judge",
            "judge_reward_weights": {"reasoning_len": 0.5},
            "max_reasoning_tokens_to_reward": 1000,
            "batch_size": 2,
            "group_size": 3,
        },
        {"num_steps": 5, "seed": 123, "max_tokens": 12000},
        _DummyRenderer(),
    )

    assert kwargs["reward_params"].judge_reward_weights == {"reasoning_len": 0.5}
    assert kwargs["reward_params"].max_reasoning_tokens_to_reward == 1000
    assert kwargs["reward_params"].reasoning_tokenizer is _DummyRenderer.tokenizer


def test_single_turn_dataset_kwargs_rejects_legacy_goal_file_key() -> None:
    """Pre-rename config keys fail loudly with the new key name in the error."""
    with pytest.raises(ValueError, match="'goal_file' in env config was renamed; use 'spec_file'"):
        single_turn_dataset_kwargs(
            {
                "judge_model": "mock/judge",
                "goal_file": "src/specs/example_character.txt",
                "batch_size": 2,
                "group_size": 3,
            },
            {"num_steps": 5, "seed": 123},
            _DummyRenderer(),
        )


def test_single_turn_dataset_kwargs_rejects_legacy_keys_in_shared_config() -> None:
    with pytest.raises(ValueError, match="'goal_reward_weights' in shared config was renamed; use 'judge_reward_weights'"):
        single_turn_dataset_kwargs(
            {
                "judge_model": "mock/judge",
                "batch_size": 2,
                "group_size": 3,
            },
            {"num_steps": 5, "seed": 123, "goal_reward_weights": {"ok": 1.0}},
            _DummyRenderer(),
        )


def test_single_turn_dataset_kwargs_rejects_missing_reasoning_len_cap() -> None:
    with pytest.raises(ValueError, match="max_reasoning_tokens_to_reward must be set"):
        single_turn_dataset_kwargs(
            {
                "judge_model": "mock/judge",
                "judge_reward_weights": {"reasoning_len": 0.5},
                "batch_size": 2,
                "group_size": 3,
            },
            {"num_steps": 5, "seed": 123, "max_tokens": 12000},
            _DummyRenderer(),
        )


def test_single_turn_dataset_kwargs_rejects_negative_reasoning_len_weight() -> None:
    with pytest.raises(ValueError, match="judge_reward_weights.reasoning_len must be non-negative"):
        single_turn_dataset_kwargs(
            {
                "judge_model": "mock/judge",
                "judge_reward_weights": {"reasoning_len": -0.5},
                "max_reasoning_tokens_to_reward": 1000,
                "batch_size": 2,
                "group_size": 3,
            },
            {"num_steps": 5, "seed": 123, "max_tokens": 12000},
            _DummyRenderer(),
        )


def test_single_turn_dataset_kwargs_rejects_reasoning_len_cap_above_rl_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        single_turn_dataset_kwargs(
            {
                "judge_model": "mock/judge",
                "judge_reward_weights": {"reasoning_len": 0.5},
                "max_reasoning_tokens_to_reward": 1001,
                "batch_size": 2,
                "group_size": 3,
            },
            {"num_steps": 5, "seed": 123, "max_tokens": 1000},
            _DummyRenderer(),
        )


@pytest.mark.asyncio
async def test_rlaif_env_passes_reasoning_once_to_score_response() -> None:
    env = RLAIFEnv(
        user_prompt="prompt",
        renderer=_DummyRenderer(),
        reward_params=RewardParameters(
            judge_model="mock/judge",
            judge_reward_weights={"reasoning_len": 0.5},
            max_reasoning_tokens_to_reward=1000,
        ),
    )
    score_response = AsyncMock(
        return_value=(1.0, {"reasoning_plan": 8, "reasoning_len": 5.0}, [], [])
    )

    with patch("src.train.rlaif.env.score_response", score_response):
        reward, metrics = await env.compute_reward("reasoning", "answer")

    assert reward == 1.0
    assert metrics["reasoning_plan"] == 8
    assert metrics["reasoning_len"] == 5.0
    assert score_response.call_args.kwargs["reasoning"] == "reasoning"
    assert "reasoning_token_count" not in score_response.call_args.kwargs
