import json

import pytest
import yaml

from src.train import pipeline
from src.train.pipeline import validate_lora_rank_matches_previous_stage


def _write_previous_stage(tmp_path, config):
    run_dir = tmp_path / "previous"
    run_dir.mkdir()
    (run_dir / "config.yml").write_text(yaml.safe_dump(config))
    return str(run_dir)


def test_validate_lora_rank_matches_previous_stage_accepts_matching_rank(tmp_path):
    previous_stage_path = _write_previous_stage(tmp_path, {"lora_rank": 8})

    validate_lora_rank_matches_previous_stage(
        {"lora_rank": 8},
        previous_stage_path,
        stage_index=1,
    )


def test_validate_lora_rank_matches_previous_stage_rejects_mismatched_rank(tmp_path):
    previous_stage_path = _write_previous_stage(tmp_path, {"lora_rank": 8})

    with pytest.raises(ValueError, match="lora_rank=32 does not match previous stage lora_rank=8"):
        validate_lora_rank_matches_previous_stage(
            {"lora_rank": 32},
            previous_stage_path,
            stage_index=1,
        )


def _write_pipeline_config(tmp_path, config):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return str(config_path)


def _patch_training_methods(monkeypatch):
    calls = []

    def run_training(config, log_path, load_checkpoint_path):
        calls.append((config, log_path, load_checkpoint_path))
        with open(f"{log_path}/checkpoints.jsonl", "w") as f:
            f.write(json.dumps({
                "name": "final",
                "state_path": f"tinker://state/{len(calls)}",
                "sampler_path": f"tinker://sampler/{len(calls)}",
            }) + "\n")

    def summarize(_config):
        return "mock-data"

    monkeypatch.setattr(
        pipeline,
        "TRAINING_METHODS",
        {"mock": (run_training, summarize)},
    )
    return calls


def _base_pipeline_config():
    return {
        "default": {
            "training_method": "mock",
            "model_name": "mock/model",
            "lora_rank": 8,
        },
        "stages": [
            {"learning_rate": 1e-4},
            {"learning_rate": 2e-4},
        ],
    }


def test_run_pipeline_chains_stages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training_calls = _patch_training_methods(monkeypatch)

    log_paths = pipeline.run_pipeline(
        exp_name="test-exp",
        config_path=_write_pipeline_config(tmp_path, _base_pipeline_config()),
        seed=123,
    )

    assert len(training_calls) == 2
    assert len(log_paths) == 2
    # The second stage resumes from the first stage's final checkpoint.
    assert training_calls[0][2] is None
    assert training_calls[1][2] == "tinker://state/1"


# ---------------------------------------------------------------------------
# cfg_to_name: crash-resume run naming (STABLE_RUN_DIR + rolling-key exclusion)
# ---------------------------------------------------------------------------

_NAME_CFG = {
    "model_name": "Qwen/Qwen3.5-4B",
    "training_method": "rl",
    "seed": 1,
    "num_steps": 10,
}


def _name(cfg):
    return pipeline.cfg_to_name(cfg, summarize_fn=lambda _: "label")


def test_cfg_to_name_stable_run_dir_drops_timestamp(monkeypatch):
    """STABLE_RUN_DIR=1 must produce the SAME run dir across launches so the
    tinker trainer's auto-resume finds the prior checkpoint."""
    monkeypatch.setenv("STABLE_RUN_DIR", "1")
    name = _name(dict(_NAME_CFG))
    assert name.endswith("-stable")
    assert name == _name(dict(_NAME_CFG))  # deterministic across "launches"

    monkeypatch.delenv("STABLE_RUN_DIR")
    assert not _name(dict(_NAME_CFG)).endswith("-stable")  # local runs unchanged


def test_cfg_to_name_rolling_keys_do_not_change_hash(monkeypatch):
    """Checkpoint cadence must be tunable mid-experiment without orphaning the
    run dir: rolling_save_every / rolling_ttl_seconds are hash-excluded."""
    monkeypatch.setenv("STABLE_RUN_DIR", "1")
    base = _name(dict(_NAME_CFG))
    tuned = _name(dict(_NAME_CFG, rolling_save_every=1, rolling_ttl_seconds=86400))
    assert base == tuned

    # Sanity: a key that IS hashed does change the name.
    assert _name(dict(_NAME_CFG, num_steps=300)) != base
