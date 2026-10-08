# Modified from tinker-cookbook tinker_cookbook/eval/inspect_evaluators.py (Apache-2.0, Thinking Machines Lab).
# See THIRD_PARTY_NOTICES.md.
"""Inspect-ai evaluator using Tinker sampling client."""

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import chz
import tinker
from inspect_ai._display.core import active as inspect_display_active
from inspect_ai import eval_async
from inspect_ai.model import ChatMessageSystem, GenerateConfig, Model, ModelOutput
from inspect_ai._util.notgiven import NOT_GIVEN
from inspect_ai.util._display import init_display_type
from tinker_cookbook.eval.evaluators import EvaluatorBuilder, SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.evals.common.dataset import get_system_prompt
from src.specs.spec import parse_spec_file
from src.utils.config import reject_legacy_config_keys
from src.tinker_local.tinker_sampling import TinkerSampler
from src.utils.system_prompt import (
    load_system_prompt_suffixes,
    resolve_system_prompt_with_suffixes,
)

logger = logging.getLogger(__name__)

# Default total token budget (prompt + output) for evals that don't set max_tokens.
TINKER_DEFAULT_MAX_TOKENS = 32768

def set_eval_params_max_tokens_if_missing(eval_params: list[dict], default_max_tokens: int) -> int:
    """Set max_tokens in-place for eval params that don't already specify it."""
    for params in eval_params:
        if "max_tokens" in params:
            continue
        params["max_tokens"] = default_max_tokens


class GenerationTruncatedError(RuntimeError):
    """Raised when a model generation is truncated by the max_tokens limit."""


class SystemMessageModel(Model):
    """Model wrapper that concatenates a system message with any existing eval system message."""

    def __init__(
        self,
        api,
        config,
        system_message: str | None = None,
        system_prompt_suffixes: list[str] | None = None,
    ):
        super().__init__(api=api, config=config)
        self._system_message = system_message
        self._system_prompt_suffixes = system_prompt_suffixes
        self._suffix_rng = random.Random(None)

    def _resolve_system_message(self) -> str | None:
        _, merged = resolve_system_prompt_with_suffixes(
            self._system_message,
            self._system_prompt_suffixes,
            rng=self._suffix_rng,
        )
        return merged

    async def generate(self, input, tools=[], tool_choice=None, config=GenerateConfig(), cache=NOT_GIVEN) -> ModelOutput:
        resolved_system_message = self._resolve_system_message()
        # `is not None` (not truthiness): an explicit empty system prompt ("") is
        # prepended as an empty system message so the renderer does NOT fall back to
        # its default identity prompt. Unset prompts resolve to None
        # and keep the renderer default.
        if resolved_system_message is not None and isinstance(input, list):
            if input and input[0].role == "system":
                merged = "\n".join(p for p in [resolved_system_message, input[0].content] if p)
                input = [ChatMessageSystem(content=merged)] + input[1:]
            else:
                input = [ChatMessageSystem(content=resolved_system_message)] + input
        output = await super().generate(input, tools, tool_choice, config, cache)
        for choice in output.choices:
            if choice.stop_reason == "max_tokens":
                tokens = output.usage.output_tokens if output.usage else "unknown"
                raise GenerationTruncatedError(
                    f"Generation truncated at max_tokens limit ({tokens} output tokens)"
                )
        return output

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_prefills_from_file(prefill_file: str, model_name: str | None = None) -> list[str]:
    """Load assistant prefills from a JSONL file, stripping EoT token if model specified."""
    path = PROJECT_ROOT / prefill_file if not Path(prefill_file).is_absolute() else Path(prefill_file)
    eos_token = get_tokenizer(model_name).eos_token if model_name else None

    prefills = []
    for line in open(path):
        for msg in json.loads(line).get("messages", []):
            if msg.get("role") == "assistant":
                content = msg["content"].replace(eos_token, "") if eos_token else msg["content"]
                prefills.append(content)
    return prefills


def load_prefills(
    prefill: str | None = None,
    prefill_file: str | None = None,
    model_name: str | None = None,
) -> list[str] | None:
    """Load prefills from file or single prefill string."""
    if prefill_file:
        prefills = load_prefills_from_file(prefill_file, model_name)
        logger.info(f"Loaded {len(prefills)} prefills from {prefill_file}")
        return prefills
    elif prefill is not None:
        return [prefill]
    return None


async def resolve_model(
    service_client: tinker.ServiceClient, model_path: str
) -> tuple[str, str | None]:
    """Resolve model_path to (base_model, tinker_path)."""
    base_model, tinker_path = parse_model_path(model_path)

    if tinker_path and not base_model:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(tinker_path)
        base_model = training_run.base_model

    return base_model, tinker_path


def parse_model_path(model_path: str) -> tuple[str | None, str | None]:
    """Parse model_path into (base_model_name, tinker_path or None)."""
    if model_path.startswith("tinker://"):
        return None, model_path
    return model_path, None


def parse_step_from_tinker_path(tinker_path: str) -> str | None:
    """Extract checkpoint step from a tinker sampler_weights path."""
    marker = "/sampler_weights/"
    if marker not in tinker_path:
        return None
    step = tinker_path.rsplit(marker, 1)[1].split("/", 1)[0]
    if step.isdigit():
        return step
    return None


def get_display_name(
    model_name: str,
    model_path: str | None = None,
) -> str:
    """Get display name like 'Qwen/Qwen3-8B@21e5262b:step000010'."""
    _, tinker_path = parse_model_path(model_path or model_name)
    if tinker_path is None:
        return model_name
    run_ref = tinker_path.removeprefix("tinker://").split("/", 1)[0]
    run_id = run_ref.split(":", 1)[0]
    name = f"{model_name}@{run_id[:8]}"
    if step := parse_step_from_tinker_path(tinker_path):
        name += f":step{step}"
    return name


def extract_metrics(results: list, eval_name: str = "") -> dict[str, float]:
    """Extract metrics from inspect-ai eval results."""
    metrics = {}
    for task_result in results:
        if task_result.results and task_result.results.scores:
            for score in task_result.results.scores:
                for name, metric in score.metrics.items():
                    prefix = f"{eval_name}/" if eval_name else ""
                    metrics[f"{prefix}{score.name or 'score'}/{name}"] = metric.value
    return metrics


# inspect_ai's eval_async does not support concurrent calls, so we serialize them.
_eval_async_lock = asyncio.Lock()


def _force_plain_inspect_display() -> None:
    """Avoid Inspect's Textual display in embedded async training evals."""
    init_display_type("plain")
    inspect_display_active._active_display = None


@chz.chz
class InspectEvaluator(SamplingClientEvaluator):
    """Runs inspect-ai task with a Tinker sampling client."""

    task: str
    renderer_name: str
    eval_name: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    prefill: str | None = None
    prefill_file: str | None = None
    limit: int | None = None
    epochs: int | None = None
    task_args: dict[str, Any] | None = None
    system_message: str | None = None
    system_prompt_suffixes: list[str] | None = None
    # Sampling params
    temperature: float = 0.6
    max_tokens: int = 32768  # Total context window budget (prompt + output); output tokens = max_tokens - prompt_tokens
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    num_choices: int = 1
    # Eval settings
    log_dir: str | None = None
    sandbox: str = "docker"
    max_connections: int = 512
    verbose: bool = False
    retry_on_error: int = 0
    debug_errors: bool = True
    log_level: str = "INFO"
    metadata: dict[str, Any] | None = None
    model_roles: dict[str, str] | None = None

    async def __call__(self, sampling_client: Any) -> dict[str, float]:
        if self.model_name is None:
            raise ValueError("model_name must be set")

        prefills = load_prefills(self.prefill, self.prefill_file, self.model_name)
        display_name = get_display_name(self.model_name, self.model_path)

        api = TinkerSampler(
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            sampling_client=sampling_client,
            verbose=self.verbose,
            display_name=display_name,
            prefills=prefills,
        )
        async with _eval_async_lock:
            _force_plain_inspect_display()
            results = await eval_async(
                tasks=self.task,
                model=[SystemMessageModel(
                    api=api,
                    config=GenerateConfig(
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        seed=self.seed,
                        num_choices=self.num_choices,
                    ),
                    system_message=self.system_message,
                    system_prompt_suffixes=self.system_prompt_suffixes,
                )],
                model_roles=self.model_roles,
                sandbox=self.sandbox,
                task_args=self.task_args or {},
                limit=self.limit,
                epochs=self.epochs,
                debug_errors=self.debug_errors,
                retry_on_error=self.retry_on_error,
                fail_on_error=False,
                log_dir=self.log_dir or os.path.expanduser("~/inspect-logs"),
                max_connections=self.max_connections,
                max_sandboxes=3 * (os.cpu_count() or 1) if self.sandbox == "docker" else None,
                log_level=self.log_level,
                log_realtime=True,
                log_buffer=1000,
                metadata=self.metadata,
            )
        task_name = self.task.removeprefix("inspect_evals/")
        eval_name = self.eval_name or (self.task_args.get("eval_name") if self.task_args else None) or task_name
        metrics = extract_metrics(results, eval_name)
        logger.info(f"Eval completed: {metrics}")
        return metrics


EVAL_META_FIELDS = {
    "task",
    "eval_name",
    "prefill",
    "prefill_file",
    "limit",
    "max_tokens",
    "top_p",
    "epochs",
    "temperature",
    "retry_on_error",
    "verbose",
    "metadata",
    "system_prompt_id",
    "sys_prompt_additions",
    "system_prompt_suffix_file",
    "prompts_dir_name",
    "dataset_file",
    "sys_prompt_model_name",
    "sandbox",
    "model_name",
    "model_path",
    "model_roles",
}


def _resolve_sys_prompt_model_name(params: dict) -> str:
    """Resolve sys_prompt_model_name from params or from the spec_file if specified."""
    if sys_prompt_model_name := params.get("sys_prompt_model_name", ""):
        return sys_prompt_model_name
    if spec_file := params.get("spec_file"):
        spec_path = PROJECT_ROOT / spec_file if not Path(spec_file).is_absolute() else Path(spec_file)
        return parse_spec_file(spec_path).get("model_name", "")
    return ""


def get_inspect_eval_builders(
    renderer_name: str,
    model_name: str,
    log_dir: str,
    eval_params: list[dict],
    model_path: str | None = None,
) -> list[EvaluatorBuilder]:
    """Build Inspect evaluator builders from eval_params list."""
    builders = []
    for params in eval_params:
        reject_legacy_config_keys(params, "eval params")
        task = params.get("task", "src/evals/common/task.py@eval_task")
        prefill = params.get("prefill")
        prefill_file = params.get("prefill_file")
        limit = params.get("limit")
        epochs = params.get("epochs")
        max_tokens = params["max_tokens"]  # Total context window budget (prompt + output)
        eval_top_p = params.get("top_p", 1.0)
        eval_temperature = params.get("temperature", 0.6)
        eval_sandbox = params.get("sandbox", "docker")
        retry_on_error = params.get("retry_on_error", 0)
        verbose = params.get("verbose", False)
        max_connections = 512
        metadata = params
        sys_prompt_model_name = _resolve_sys_prompt_model_name(params)
        if "system_prompt_id" in params:
            system_message = get_system_prompt(
                system_prompt_id=params["system_prompt_id"],
                prompts_dir_name=params.get("prompts_dir_name", "general"),
                sys_prompt_additions=params.get("sys_prompt_additions"),
                sys_prompt_model_name=sys_prompt_model_name,
            )
        else:
            system_message = None
        system_prompt_suffixes = None
        if system_prompt_suffix_file := params.get("system_prompt_suffix_file"):
            system_prompt_suffixes = [s["content"] for s in load_system_prompt_suffixes(system_prompt_suffix_file)]
        task_args = {k: v for k, v in params.items() if k not in EVAL_META_FIELDS}
        eval_name = params.get("eval_name", task)
        # Re-add eval_task-specific params that were stripped as meta fields
        if "@eval_task" in task:
            if sys_prompt_model_name:
                task_args["sys_prompt_model_name"] = sys_prompt_model_name
            task_args["prompts_dir_name"] = params["prompts_dir_name"]
            task_args["eval_name"] = eval_name
            if dataset_file := params.get("dataset_file"):
                task_args["dataset_file"] = dataset_file
        eval_log_dir = f"{log_dir}/evals/{eval_name}"
        eval_model_roles = params.get("model_roles")
        def builder(task=task, task_args=task_args, eval_log_dir=eval_log_dir,
                    prefill=prefill, prefill_file=prefill_file, limit=limit,
                    epochs=epochs, max_tokens=max_tokens, system_message=system_message,
                    system_prompt_suffixes=system_prompt_suffixes,
                    eval_top_p=eval_top_p, eval_temperature=eval_temperature, eval_sandbox=eval_sandbox,
                    retry_on_error=retry_on_error, verbose=verbose,
                    max_connections=max_connections,
                    eval_name=eval_name, metadata=metadata,
                    eval_model_roles=eval_model_roles) -> InspectEvaluator:
            return InspectEvaluator(
                task=task,
                eval_name=eval_name,
                task_args=task_args,
                renderer_name=renderer_name,
                model_name=model_name,
                model_path=model_path,
                prefill=prefill,
                prefill_file=prefill_file,
                limit=limit,
                epochs=epochs,
                system_message=system_message,
                system_prompt_suffixes=system_prompt_suffixes,
                temperature=eval_temperature,
                top_p=eval_top_p,
                max_tokens=max_tokens,
                sandbox=eval_sandbox,
                retry_on_error=retry_on_error,
                verbose=verbose,
                debug_errors=True,
                log_dir=eval_log_dir,
                max_connections=max_connections,
                log_level="INFO",
                metadata=metadata,
                model_roles=eval_model_roles,
            )
        builders.append(builder)
    return builders
