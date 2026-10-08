"""RL training runner.

Reads an `envs` list from the config, each specifying a type (the Corin runs use a single
"reward_hack" env; see src/train/rlaif/reward_hack/README.md) and its parameters. Multiple
envs are mixed into a single training batch.

To add a new env type:
1. Implement the env + dataset in a new module
2. Add a build function to ENV_BUILDERS below
"""

import asyncio
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import chz
import tinker
from dotenv import load_dotenv
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.tinker_local.inspect_evaluators import (
    TINKER_DEFAULT_MAX_TOKENS,
    get_inspect_eval_builders,
    set_eval_params_max_tokens_if_missing,
)
from src.tinker_local.tinker_sampling import get_renderer, get_renderer_name_for_model
from src.train.rlaif.env import build_rlaif_dataset
from src.train.rlaif.mixed_dataset import MixedRLDataset
from src.train.rlaif.reward_hack.env import build_reward_hack_dataset
from src.train.rlaif.reward_hack.heldout_eval import build_heldout_reward_hack_evaluator
from src.train.rlaif.sample_logger import step_sample_logger

logger = logging.getLogger(__name__)


def _make_renderer(shared: dict):
    """Create renderer from shared config (common to all env types)."""
    model_name = shared["model_name"]
    renderer_name = shared.get("renderer_name") or get_renderer_name_for_model(model_name)
    tokenizer = get_tokenizer(model_name)
    return get_renderer(renderer_name, tokenizer)


# Registry: env type name → builder function (env_cfg, shared, renderer) -> RLDataset
ENV_BUILDERS = {
    "rlaif": build_rlaif_dataset,
    "reward_hack": build_reward_hack_dataset,
}


# ---------------------------------------------------------------------------
# Dataset builder that wraps multiple env datasets
# ---------------------------------------------------------------------------


@chz.chz
class MixedRLDatasetBuilder(RLDatasetBuilder):
    """Builds a MixedRLDataset from a list of env configs."""

    envs_config: list[dict]
    shared_config: dict

    async def __call__(self) -> tuple[MixedRLDataset, None]:
        renderer = _make_renderer(self.shared_config)
        sub_datasets = []
        for env_cfg in self.envs_config:
            env_type = env_cfg["type"]
            if env_type not in ENV_BUILDERS:
                raise ValueError(f"Unknown env type '{env_type}'. Available: {list(ENV_BUILDERS)}")
            builder_fn = ENV_BUILDERS[env_type]
            ds = builder_fn(env_cfg, self.shared_config, renderer)
            logger.info(f"Built {env_type} env dataset: batch_size={env_cfg['batch_size']}, {len(ds)} steps")
            sub_datasets.append(ds)

        total_batch = sum(env_cfg["batch_size"] for env_cfg in self.envs_config)
        logger.info(f"Mixed dataset: {len(sub_datasets)} env types, total batch_size={total_batch}")

        return MixedRLDataset(
            sub_datasets=sub_datasets,
            num_steps=self.shared_config["num_steps"],
        ), None


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


@contextmanager
def _override_lora_modules(train_mlp: bool, train_unembed: bool):
    """Temporarily patch ServiceClient to pass train_mlp/train_unembed to LoRA client creation."""
    orig = tinker.ServiceClient.create_lora_training_client_async
    async def patched(self, *args, **kwargs):
        kwargs.setdefault("train_mlp", train_mlp)
        kwargs.setdefault("train_unembed", train_unembed)
        return await orig(self, *args, **kwargs)
    tinker.ServiceClient.create_lora_training_client_async = patched
    try:
        yield
    finally:
        tinker.ServiceClient.create_lora_training_client_async = orig


@contextmanager
def _override_weight_decay(weight_decay: float):
    """Patch AdamParams to inject weight_decay into all RL optimizer steps.

    tinker_cookbook.rl.train constructs AdamParams without weight_decay;
    this injects it so we don't need to modify the installed package.
    """
    OrigAdamParams = tinker.AdamParams
    def patched(*args, **kwargs):
        kwargs.setdefault("weight_decay", weight_decay)
        return OrigAdamParams(*args, **kwargs)
    tinker.AdamParams = patched
    try:
        yield
    finally:
        tinker.AdamParams = OrigAdamParams


def _prefix_true_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Move unfiltered trajectory metrics from env/... to env_true/... keys."""
    return {
        key.replace("env/", "env_true/", 1): value
        for key, value in metrics.items()
        if key.startswith("env/")
    }


def _compute_true_rollout_metrics(groups) -> dict[str, Any]:
    trajectory_groups = [group.trajectory_group for group in groups]
    taglists = [group.tags for group in groups]
    return _prefix_true_metrics(
        rl_train.compute_trajectory_metrics(trajectory_groups, taglists)
    )


@contextmanager
def _log_true_rollout_metrics():
    """Log unfiltered rollout metrics alongside the filtered training metrics."""
    orig_export = rl_train._maybe_export_rollout_summary_jsonl
    orig_setup_logging = rl_train.ml_log.setup_logging
    pending_metrics_by_step: dict[int, dict[str, Any]] = defaultdict(dict)

    def patched_export(*args, **kwargs):
        split = kwargs.get("split")
        base_name = kwargs.get("base_name")
        iteration = kwargs.get("iteration")
        groups = kwargs.get("groups_P") or []
        if split == "train" and base_name == "train" and iteration is not None and groups:
            pending_metrics_by_step[int(iteration)].update(
                _compute_true_rollout_metrics(groups)
            )
        return orig_export(*args, **kwargs)

    def patched_setup_logging(*args, **kwargs):
        ml_logger = orig_setup_logging(*args, **kwargs)
        orig_log_metrics = ml_logger.log_metrics

        def patched_log_metrics(metrics, step):
            true_metrics = pending_metrics_by_step.pop(step, None)
            if true_metrics:
                metrics.update(true_metrics)
            return orig_log_metrics(metrics, step)

        ml_logger.log_metrics = patched_log_metrics
        return ml_logger

    rl_train._maybe_export_rollout_summary_jsonl = patched_export
    rl_train.ml_log.setup_logging = patched_setup_logging
    try:
        yield
    finally:
        rl_train._maybe_export_rollout_summary_jsonl = orig_export
        rl_train.ml_log.setup_logging = orig_setup_logging


def _build_heldout_reward_hack_eval_builders(
    config: dict, log_path: str, renderer_name: str
) -> list:
    """Build held-out reward_hack evaluator builders, gated on eval_every>0.

    RESUME-HASH SAFETY: this is auto-added whenever ``eval_every > 0`` and a
    ``reward_hack`` env is present — it introduces NO new HASHED config key
    (``eval_every`` is already in ``cfg_to_name``'s ``keys_to_remove``). So a
    paused run resumes WITH the held-out eval on without orphaning its checkpoint.
    ``log_path`` and ``renderer_name`` are therefore plumbed as FUNCTION ARGS, not
    config: the evaluator needs them to persist its per-rollout transcripts (the
    cookbook hands evaluators neither an output dir nor a step number).

    ``renderer_name`` is the string the CALLER already resolved (a Renderer object
    does not expose the name it was built from). It is passed in rather than
    re-derived here so there is exactly one place in this module that decides it —
    a second derivation could silently drift from the renderer actually in use and
    stamp every transcript row with a lie.

    For each ``reward_hack`` env block we build one evaluator over that block's
    held-out split (same splits / heldout_frac / seed / show_test_in_prompt /
    timeout as training) scored under a FIXED neutral prompt. Defensive: any
    failure to construct an evaluator is logged and skipped, never raised.
    """
    eval_every = config.get("eval_every", 0)
    if eval_every <= 0:
        return []

    renderer = _make_renderer(config)
    builders: list = []
    for env_index, env_cfg in enumerate(config.get("envs", [])):
        if env_cfg.get("type") != "reward_hack":
            continue
        try:
            evaluator = build_heldout_reward_hack_evaluator(
                env_cfg,
                config,
                renderer,
                log_path=log_path,
                renderer_name=renderer_name,
                # Env-block index: keeps transcript rows attributable if a config
                # ever declares more than one reward_hack block.
                evaluator_id=env_index,
            )
        except Exception as exc:
            logger.warning("Failed to build held-out reward_hack evaluator: %r", exc)
            continue
        if evaluator is None:
            continue
        # Plain builder callable: the cookbook trainer calls builder() -> evaluator.
        builders.append(lambda evaluator=evaluator: evaluator)
    if builders:
        logger.info(
            "Held-out reward_hack eval ENABLED (eval_every=%d): %d evaluator(s).",
            eval_every, len(builders),
        )
    return builders


def run_training(config: dict, log_path: str, load_checkpoint_path: str | None = None):
    """Run RL training with the given config.

    Config must have an `envs` list, each entry specifying `type` and env-specific params.
    """
    load_dotenv()

    model_name = config["model_name"]
    # Allow config to override the renderer (e.g. to disable thinking for the
    # reward-hack arm so it matches the prompted-hack harness). Defaults to the
    # model's recommended renderer when unset.
    renderer_name = config.get("renderer_name") or get_renderer_name_for_model(model_name)
    run_name = os.path.basename(log_path)
    eval_every = config.get("eval_every", 0)

    envs_config = config["envs"]
    dataset_builder = MixedRLDatasetBuilder(
        envs_config=envs_config,
        shared_config=config,
    )

    eval_params = config.get("eval_params", [])
    set_eval_params_max_tokens_if_missing(
        eval_params=eval_params,
        default_max_tokens=TINKER_DEFAULT_MAX_TOKENS,
    )

    eval_builders = get_inspect_eval_builders(
        renderer_name=renderer_name,
        model_name=model_name,
        log_dir=log_path,
        eval_params=eval_params,
    )

    # Periodic HELD-OUT reward_hack eval (deconfounded headline metric). Auto-added
    # when eval_every>0 and a reward_hack env is present; no new hashed config key,
    # so a paused run resumes with eval on (see the helper + cfg_to_name).
    eval_builders = list(eval_builders) + _build_heldout_reward_hack_eval_builders(
        config, log_path, renderer_name
    )

    rl_config = rl_train.Config(
        model_name=model_name,
        recipe_name="character_training_rlaif",
        dataset_builder=dataset_builder,
        load_checkpoint_path=load_checkpoint_path,
        renderer_name=renderer_name,
        learning_rate=config.get("learning_rate", 1e-5),
        max_tokens=config["max_tokens"],
        temperature=config.get("temperature", 1.0),
        log_path=log_path,
        lora_rank=config.get("lora_rank", 8),
        eval_every=eval_every,
        # Permanent (sampler-weight) checkpoint cadence, decoupled from eval_every so we can
        # save every RL step (save_every=1) without running the full eval suite every step.
        save_every=config.get("save_every", eval_every),
        evaluator_builders=eval_builders,
        loss_fn="importance_sampling",
        num_substeps=config.get("num_substeps", 1),
        wandb_project=config.get("wandb_project", "character-training"),
        wandb_name=config.get("wandb_name") or run_name,
        remove_constant_reward_groups=True,
        kl_penalty_coef=config.get("kl_penalty", 0.0),
        ttl_seconds=None,
        # Crash-resume cadence. rolling_save_every>0 makes the trainer
        # periodically save *resumable training STATE* (kind="state" ->
        # state_path in checkpoints.jsonl) without the expensive sampler-weight
        # export. On relaunch with the SAME log_path, rl_train.main calls
        # checkpoint_utils.get_last_checkpoint(log_path, required_key="state_path")
        # and resumes from that step. 0 (the cookbook default) disables this, so
        # a crash loses the whole run — hence we surface it as a config key.
        # rolling_ttl_seconds is the SERVER-SIDE retention of those rolling
        # state checkpoints (cookbook default 7200s = 2h); we let configs raise
        # it so a crash+relaunch gap longer than 2h doesn't lose the checkpoint.
        rolling_save_every=config.get("rolling_save_every", 0),
        rolling_ttl_seconds=config.get("rolling_ttl_seconds", 7200),
    )

    step_sample_logger.configure(
        log_path,
        samples_per_key=config.get("samples_per_key", 10),
    )

    train_mlp = config.get("train_mlp", True)
    train_unembed = config.get("train_unembed", False)
    weight_decay = config.get("weight_decay", 0.0)
    logger.info(f"Creating LoRA training client with train_mlp={train_mlp}, train_unembed={train_unembed}, weight_decay={weight_decay}")
    with (
        _override_lora_modules(train_mlp, train_unembed),
        _override_weight_decay(weight_decay),
        _log_true_rollout_metrics(),
    ):
        asyncio.run(rl_train.main(rl_config))

    step_sample_logger.finalize()


def summarize_envs(config: dict) -> str:
    """Create a short label from envs config for run naming."""
    envs = config.get("envs", [])
    if not envs:
        return "no-envs"
    return "+".join(env_cfg["type"] for env_cfg in envs)
