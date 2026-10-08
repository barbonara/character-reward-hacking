"""RLAIF environment classes for RL training with LLM judge rewards."""

import json
import logging
import random
from abc import abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers.base import Message, Renderer, ToolSpec
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    StepResult,
)
from tinker_cookbook.utils import logtree

from src.train.rlaif.llm_judge import JudgeTrace, RewardParameters, score_response
from src.train.rlaif.sample_logger import step_sample_logger
from src.utils.config import reject_legacy_config_keys
from src.utils.parsing import parse_action_to_reasoning_and_response
from src.utils.system_prompt import load_system_prompt_suffixes, resolve_system_prompt_with_suffixes

logger = logging.getLogger(__name__)


def load_prompts_from_file(path: Path) -> list[str]:
    """Load prompts from a file. For JSONL, extracts user messages from chat format."""
    lines = path.read_text().splitlines()
    if path.suffix != ".jsonl":
        return [line.strip() for line in lines if line.strip()]

    prompts = []
    for line in lines:
        try:
            for msg in json.loads(line)["messages"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"])
        except (json.JSONDecodeError, KeyError):
            pass
    return prompts


def _split_reward_prompt_path(env_cfg: dict) -> tuple[str | None, dict[str, str] | None]:
    reward_prompt_path = env_cfg.get("reward_prompt_path")
    if isinstance(reward_prompt_path, dict):
        return None, reward_prompt_path
    return reward_prompt_path, None


def _reasoning_len_weight(env_cfg: dict) -> float:
    return (env_cfg.get("judge_reward_weights") or {}).get("reasoning_len", 0.0)


def _validate_reasoning_len_reward(env_cfg: dict, shared: dict) -> None:
    cap = env_cfg.get("max_reasoning_tokens_to_reward")
    weight = _reasoning_len_weight(env_cfg)
    if weight < 0:
        raise ValueError("judge_reward_weights.reasoning_len must be non-negative")
    if weight > 0 and cap is None:
        raise ValueError("max_reasoning_tokens_to_reward must be set when reasoning_len is rewarded")
    if cap is not None and weight == 0:
        raise ValueError(
            "judge_reward_weights.reasoning_len must be set when "
            "max_reasoning_tokens_to_reward is configured (no default weights)"
        )
    if cap is not None and not 0 < cap <= shared["max_tokens"]:
        raise ValueError("max_reasoning_tokens_to_reward must be in (0, max_tokens]")


def single_turn_dataset_kwargs(env_cfg: dict, shared: dict, renderer: Renderer) -> dict:
    """Build common ``SingleTurnDataset`` kwargs from an env config."""
    reject_legacy_config_keys(env_cfg, "env config")
    reject_legacy_config_keys(shared, "shared config")
    base_reward_prompt_path, reward_prompt_paths = _split_reward_prompt_path(env_cfg)
    sys_prompt_suffixes = None
    if suffix_file := env_cfg.get("system_prompt_suffix_file"):
        sys_prompt_suffixes = load_system_prompt_suffixes(suffix_file)
    _validate_reasoning_len_reward(env_cfg, shared)

    reward_params = RewardParameters(
        judge_model=env_cfg["judge_model"],
        reasoning_effort=env_cfg.get("reasoning_effort"),
        skip_no_spec_mention=env_cfg.get("skip_no_spec_mention", False),
        spec_file=env_cfg.get("spec_file", shared.get("spec_file")),
        reward_prompt_path=base_reward_prompt_path,
        judge_reward_weights=env_cfg.get("judge_reward_weights"),
        reasoning_plan_separate_call=env_cfg.get("reasoning_plan_separate_call", False),
        max_reasoning_tokens_to_reward=env_cfg.get("max_reasoning_tokens_to_reward"),
        reasoning_tokenizer=renderer.tokenizer,
    )
    return {
        "renderer": renderer,
        "batch_size": env_cfg["batch_size"],
        "group_size": env_cfg["group_size"],
        "num_steps": shared["num_steps"],
        "sys_prompt": env_cfg.get("sys_prompt"),
        "sys_prompt_suffixes": sys_prompt_suffixes,
        "reward_prompt_paths": reward_prompt_paths,
        "reward_params": reward_params,
        "logging_tag": env_cfg.get("logging_tag"),
        "seed": shared.get("seed"),
    }


# ---------------------------------------------------------------------------
# Base classes shared across env types (RLAIF, math, etc.)
# ---------------------------------------------------------------------------


class SingleTurnEnv(Env):
    """Base class for single-turn RL environments using our custom renderer.

    Handles prompt rendering, response parsing (stop token + think tag splitting),
    and the common step() skeleton. Subclasses implement compute_reward().

    Subclasses must set `env_type` as a class attribute (e.g. env_type = "rlaif").
    """

    env_type: str

    def __init__(
        self,
        user_prompt: str,
        renderer: Renderer,
        sys_prompt: str | None = None,
        tool_specs: list[ToolSpec] | None = None,
        suffix_name: str | None = None,
    ):
        self.user_prompt = user_prompt
        self.renderer = renderer
        self.sys_prompt = sys_prompt
        self.tool_specs = tool_specs
        self.suffix_name = suffix_name
        self.logging_tag: str | None = None
        self._observation: Observation | None = None
        # Populated by compute_reward in subclasses; consumed by step() for HTML logging.
        self._judge_traces: list[JudgeTrace] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.tool_specs:
            # Tool declarations + system prompt are handled together by the renderer
            messages = self.renderer.create_conversation_prefix_with_tools(
                self.tool_specs, self.sys_prompt or ""
            )
        else:
            messages: list[Message] = []
            if self.sys_prompt:
                messages.append({"role": "system", "content": self.sys_prompt})
        messages.append({"role": "user", "content": self.user_prompt})
        observation = self.renderer.build_generation_prompt(messages)
        self._observation = observation
        return observation, self.stop_condition

    def _build_datum_text(self, action_tokens: list[int]) -> str:
        """Decode observation + action tokens into the full datum string."""
        tokenizer = self.renderer.tokenizer
        obs_ints = self._observation.to_ints() if self._observation else []
        return tokenizer.decode(obs_ints + action_tokens)

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        reasoning, visible_response = parse_action_to_reasoning_and_response(action, self.renderer)
        reward, metrics = await self.compute_reward(reasoning, visible_response)
        step_sample_logger.maybe_log(
            env_type=self.logging_tag or self.env_type,
            suffix_name=self.suffix_name,
            datum_text=self._build_datum_text(action),
            total_reward=reward,
            metrics=metrics,
            judge_traces=self._judge_traces,
        )
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    @abstractmethod
    async def compute_reward(self, reasoning: str, visible_response: str) -> tuple[float, dict]:
        """Compute reward from the parsed reasoning and visible response.

        Returns (reward, metrics_dict).
        """
        ...


@dataclass(frozen=True)
class SingleTurnGroupBuilder(EnvGroupBuilder):
    """Creates a group of single-turn environments using a thunk."""

    env_thunk: Callable[[], SingleTurnEnv]
    num_envs: int
    tags: list[str]

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return self.tags


class SingleTurnDataset(RLDataset):
    """Single-turn RL dataset with group-based batching.

    Combines an env class with data items to produce RL training batches.
    Supports optional system prompt suffixes with per-suffix reward params.
    """

    def __init__(
        self,
        *,
        items: list[dict],
        batch_size: int,
        group_size: int,
        num_steps: int,
        env_cls: type[SingleTurnEnv],
        tags: list[str],
        renderer: Renderer,
        reward_params: RewardParameters,
        sys_prompt: str | None = None,
        sys_prompt_suffixes: list[dict] | None = None,
        reward_prompt_paths: dict[str, str] | None = None,
        logging_tag: str | None = None,
        seed: int | None = None,
    ):
        self._items = items
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_steps = num_steps
        self.env_cls = env_cls
        self.tags = [logging_tag] if logging_tag else tags
        self.renderer = renderer
        self.reward_params = reward_params
        self.sys_prompt = sys_prompt
        self.sys_prompt_suffixes = sys_prompt_suffixes
        self.reward_prompt_paths = reward_prompt_paths
        self.logging_tag = logging_tag
        self._suffix_rng = random.Random(seed)

    def _resolve_reward_params(self, suffix_name: str | None) -> RewardParameters:
        """Resolve per-suffix reward params, falling back to the base reward_params."""
        if not self.reward_prompt_paths or suffix_name is None:
            return self.reward_params
        if suffix_name not in self.reward_prompt_paths:
            return self.reward_params
        return replace(self.reward_params, reward_prompt_path=self.reward_prompt_paths[suffix_name])

    def _make_group_builder(self, batch_index: int, item_index: int, item) -> SingleTurnGroupBuilder:
        suffix_name, chosen_sys_prompt = resolve_system_prompt_with_suffixes(
            self.sys_prompt,
            self.sys_prompt_suffixes,
            rng=self._suffix_rng,
        )
        reward_params = self._resolve_reward_params(suffix_name)
        logger.info(f"Batch {batch_index} sample {item_index}: suffix={suffix_name}, reward_prompt={reward_params.reward_prompt_path}")
        def env_thunk() -> SingleTurnEnv:
            env = self.env_cls(
                renderer=self.renderer,
                sys_prompt=chosen_sys_prompt,
                suffix_name=suffix_name,
                reward_params=reward_params,
                **item,
            )
            env.logging_tag = self.logging_tag
            return env

        return SingleTurnGroupBuilder(env_thunk=env_thunk, num_envs=self.group_size, tags=self.tags)

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            self._make_group_builder(index, i, self._items[(index * self.batch_size + i) % len(self._items)])
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.num_steps


# ---------------------------------------------------------------------------
# RLAIF environment
# ---------------------------------------------------------------------------


class RLAIFEnv(SingleTurnEnv):
    """Single-turn environment that computes reward via LLM judge."""

    env_type = "rlaif"

    def __init__(
        self,
        user_prompt: str,
        renderer: Renderer,
        reward_params: RewardParameters,
        sys_prompt: str | None = None,
        suffix_name: str | None = None,
    ):
        super().__init__(user_prompt, renderer, sys_prompt, suffix_name=suffix_name)
        self.reward_params = reward_params

    async def compute_reward(self, reasoning: str, visible_response: str) -> tuple[float, dict]:
        metrics: dict = {}
        if not reasoning or not visible_response:
            logtree.log_text(f"Prompt: {self.user_prompt}")
            logtree.log_text(f"Missing {'reasoning' if not reasoning else 'response'}, reward=0")
            return 0.0, metrics

        reward, scores, _, traces = await score_response(
            question=self.user_prompt,
            reasoning=reasoning,
            response=visible_response,
            reward_params=self.reward_params,
        )
        self._judge_traces.extend(traces)
        if scores:
            metrics.update(scores)
        logtree.log_text(f"Reward: {reward:.3f}")
        return reward, metrics


def build_rlaif_dataset(env_cfg: dict, shared: dict, renderer: Renderer) -> SingleTurnDataset:
    """Build an RLAIF dataset from env config."""
    data_files = env_cfg["data_files"]
    prompts = []
    for f in data_files:
        prompts.extend(load_prompts_from_file(Path(f)))
    seed = shared.get("seed")
    random.Random(seed).shuffle(prompts)
    logger.info(f"RLAIF env: loaded {len(prompts)} prompts from {len(data_files)} files")

    items = [{"user_prompt": p} for p in prompts]
    return SingleTurnDataset(
        items=items,
        env_cls=RLAIFEnv,
        tags=["rlaif"],
        **single_turn_dataset_kwargs(env_cfg, shared, renderer),
    )
