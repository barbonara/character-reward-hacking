"""Pipeline for running sequential training stages with checkpoint chaining.

This module runs multiple training stages sequentially, where each stage after
the first automatically resumes from the previous stage's final checkpoint.

Usage:
    uv run python -m src.train.pipeline --exp_name my_exp --config config.yaml --seed 1

Config format (YAML):
    default:
      <default hparams>

    stages:
      - training_method: sft
        <Stage specific hparams>

      - training_method: sft
        <Stage specific hparams>

Stage values override default values.
`previous_stage_path` can be given the path to a local dir to train starting from that stage's model; for subsequent stages it's set automatically.
`previous_stage_step` can optionally specify which checkpoint step to use from previous_stage_path (defaults to last checkpoint).

The pipeline creates a new dir for each stage in the pipeline. The dir name includes basic information as well as the config hash, seed, and timestamp. The hash itself ignores the seed, but the hash will depend on the seed of the previous stage.

Training methods:
    Each stage must specify a `training_method` field (or inherit from default).
    Supported methods:
      - 'sft': Supervised fine-tuning (src/train/sft.py)
      - 'rl': RL training with extensible envs list (src/train/rlaif/train.py)
    Add to TRAINING_METHODS to support more training methods.
"""

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv

from src.utils.config import normalize_config_for_grouping
from src.train import sft
from src.train.rlaif import train as rlaif
from src.utils import deep_merge

# Mapping from training_method name to (run_function, summarize_data_files_function)
TRAINING_METHODS: dict[str, tuple[Callable, Callable]] = {
    "sft": (sft.run_training, sft.summarize_data_files),
    "rl": (rlaif.run_training, rlaif.summarize_envs),
}

# Load .env from repo root at module load time
load_dotenv()


def load_pipeline_config(config_path: str) -> dict:
    """Load and validate pipeline config from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if "default" not in config:
        raise ValueError("Config must have a 'default' section")
    if "stages" not in config or not config["stages"]:
        raise ValueError("Config must have at least one stage in 'stages'")
    
    return config


def get_checkpoint_from_path(
    log_path: str, step: int | None = None, path_key: str = "state_path"
) -> str:
    """Read checkpoints.jsonl and return a checkpoint path.

    Args:
        log_path: Path to the training run directory, or a tinker:// URI (returned as-is).
        step: If provided, return the checkpoint at this step. If None, return the last checkpoint.
        path_key: Which path to return from the checkpoint entry (e.g. "state_path", "sampler_path").
    """
    if log_path.startswith("tinker://"):
        return log_path
    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        raise FileNotFoundError(f"No checkpoints.jsonl found in {log_path}")
    
    checkpoints = []
    with open(checkpoints_file) as f:
        for line in f:
            line = line.strip()
            if line:
                checkpoints.append(json.loads(line))
    
    if not checkpoints:
        raise ValueError(f"checkpoints.jsonl is empty in {log_path}")
    
    if step is None:
        ckpt = checkpoints[-1]
    else:
        ckpt = None
        for candidate in checkpoints:
            name = candidate.get("name")
            try:
                ckpt_step = int(name)
            except (ValueError, TypeError):
                continue
            if ckpt_step == step:
                ckpt = candidate
                break
        if ckpt is None:
            available = [c.get("name") for c in checkpoints]
            raise ValueError(f"No checkpoint found at step {step} in {log_path}. Available: {available}")
    
    if path_key not in ckpt:
        raise ValueError(
            f"Checkpoint '{ckpt.get('name')}' in {log_path} has no '{path_key}'. "
            f"Available keys: {list(ckpt.keys())}"
        )
    return ckpt[path_key]

def validate_lora_rank_matches_previous_stage(stage_config: dict, previous_stage_path: str, stage_index: int) -> None:
    if previous_stage_path.startswith("tinker://"):
        return
    with open(Path(previous_stage_path) / "config.yml") as f:
        previous_rank = yaml.safe_load(f).get("lora_rank")
    current_rank = stage_config.get("lora_rank")
    if current_rank != previous_rank:
        raise ValueError(
            f"Stage {stage_index}: lora_rank={current_rank} does not match "
            f"previous stage lora_rank={previous_rank} from {previous_stage_path}"
        )

def hash_string(s: str) -> str:
    h = hashlib.sha256(s.encode())
    b64_encoded = base64.b64encode(h.digest()).decode("ascii")
    return ("".join(filter(str.isalnum, b64_encoded)) + h.hexdigest())[:10]

def cfg_to_name(yml_cfg, summarize_fn: Callable) -> str:
    data_label = summarize_fn(yml_cfg)
    model_name_for_run = yml_cfg["model_name"].replace("/", "-")
    training_method = yml_cfg["training_method"]
    seed = yml_cfg.get("seed", None)
    
    keys_to_remove = [
        'save_every', 'eval_every', 'infrequent_eval_every',
        'wandb_project', 'wandb_name', 'log_path', 'log_dir',
        # Checkpoint cadence must not change the run dir, so how often resumable
        # state is saved can be tuned without orphaning an in-progress run's checkpoint.
        'rolling_save_every', 'rolling_ttl_seconds',
    ]
    normalized = normalize_config_for_grouping(yml_cfg, keys_to_remove)
    normalized.pop("seed", None)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    cfg_hash = hash_string(serialized)

    # Crash-resume support: with STABLE_RUN_DIR=1 the run dir is stable across
    # relaunches (no timestamp), so the tinker trainer's auto-resume
    # (checkpoint_utils.get_last_checkpoint(log_path)) finds the prior checkpoint
    # and continues. cfg_hash + seed still identify the (config, seed) pair.
    # Without the env var, every launch gets a new timestamped dir.
    if os.environ.get("STABLE_RUN_DIR") == "1":
        date_and_time = "stable"
    else:
        date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = f"{training_method}-{data_label}-{model_name_for_run}-{cfg_hash}-{seed}-{date_and_time}"

    # Linux filenames are limited to 255 bytes. Cap below that to leave room for
    # nested files (checkpoints/<step>/...). cfg_hash already provides uniqueness,
    # so truncating the data_label is safe.
    max_name_len = 200
    if len(name) > max_name_len:
        overflow = len(name) - max_name_len
        truncated_label = data_label[: max(0, len(data_label) - overflow)]
        name = f"{training_method}-{truncated_label}-{model_name_for_run}-{cfg_hash}-{seed}-{date_and_time}"
    return name


def run_pipeline(exp_name: str, config_path: str, seed: int | None = None) -> list[str]:
    """Run all training stages sequentially.
    
    Args:
        exp_name: Experiment name for the log directory.
        config_path: Path to the YAML config file.
        seed: Random seed for shuffling data.
        
    Returns:
        List of log_path directories for each completed stage.
    """
    config = load_pipeline_config(config_path)
    default_config = config["default"]
    stages = config["stages"]
    
    # Set seed in default config if provided
    if seed is not None:
        default_config["seed"] = seed
    
    # Validate previous_stage_path and previous_stage_step usage
    for i, stage in enumerate(stages):
        if i > 0 and "previous_stage_path" in stage:
            raise ValueError(
                f"Stage {i}: previous_stage_path can only be set on the first stage"
            )
        if "previous_stage_step" in stage and "previous_stage_path" not in stage:
            raise ValueError(
                f"Stage {i}: previous_stage_step requires previous_stage_path"
            )
    
    log_paths: list[str] = []
    previous_stage_path: str | None = None
    
    for i, stage_config in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Starting stage {i + 1}/{len(stages)}")
        print(f"{'='*60}\n")
        
        # Merge stage config with defaults
        merged_config = deep_merge(default_config, stage_config)
        
        # Resolve checkpoint path
        load_checkpoint_path = None
        if i == 0:
            if "previous_stage_path" in stage_config:
                validate_lora_rank_matches_previous_stage(
                    merged_config,
                    stage_config["previous_stage_path"],
                    i,
                )
                step = stage_config.get("previous_stage_step")
                load_checkpoint_path = get_checkpoint_from_path(
                    stage_config["previous_stage_path"], step=step
                )
                print(f"Loading checkpoint from: {load_checkpoint_path}")
        else:
            validate_lora_rank_matches_previous_stage(merged_config, previous_stage_path, i)
            load_checkpoint_path = get_checkpoint_from_path(previous_stage_path)
            merged_config["previous_stage_path"] = previous_stage_path
            print(f"Loading checkpoint from previous stage: {load_checkpoint_path}")
        
        merged_config["load_checkpoint_path"] = load_checkpoint_path
        
        # Get training method
        training_method = merged_config.get("training_method", "sft")
        if training_method not in TRAINING_METHODS:
            raise ValueError(
                f"Unknown training_method '{training_method}'. "
                f"Supported: {list(TRAINING_METHODS.keys())}"
            )
        run_fn, summarize_fn = TRAINING_METHODS[training_method]
        
        # Generate log path
        run_name = cfg_to_name(merged_config, summarize_fn)
        log_path = f"experiments/{exp_name}/{run_name}"
        
        # Create log directory and save config
        os.makedirs(log_path, exist_ok=True)
        config_save_path = Path(log_path) / "config.yml"
        with open(config_save_path, "w") as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
        print(f"Saved config to: {config_save_path}")
        
        # Run training
        print(f"Log path: {log_path}")
        print(f"Training method: {training_method}")
        run_fn(merged_config, log_path, load_checkpoint_path)
        
        log_paths.append(log_path)
        previous_stage_path = log_path
        
        print(f"\nCompleted stage {i + 1}/{len(stages)}")

    print(f"\n{'='*60}")
    print("Pipeline completed!")
    print(f"{'='*60}")
    print("\nLog paths:")
    for i, path in enumerate(log_paths):
        print(f"  Stage {i + 1}: {path}")
    
    return log_paths


def main():
    parser = argparse.ArgumentParser(description="Run sequential training pipeline")
    parser.add_argument(
        "--exp_name", required=True, help="Experiment name"
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for shuffling data"
    )
    args = parser.parse_args()
    
    run_pipeline(args.exp_name, args.config, seed=args.seed)


if __name__ == "__main__":
    main()
