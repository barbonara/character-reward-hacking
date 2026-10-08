"""Tests for hack-preferential per-step transcript sampling (StepSampleLogger).

The monitorability DV is the CoT of *hacked* rollouts, which are rare (~15-20%
at baseline). A uniform ~30/512 sample captured 0 hack transcripts in an early
baseline run, so the logger must retain ALL grader-flagged hacked rollouts (up
to the per-key budget) and only fill the remainder randomly. These tests drive
the logger end-to-end (through the real logtree HTML flush) with simulated
rollout metrics.
"""

import random

import pytest

from src.train.rlaif.sample_logger import _HACK_FLAG_KEYS, StepSampleLogger, _is_hacked


def _feed_rollouts(logger_obj, n_total, hacked_indices, env_type="reward_hack"):
    """Feed n_total simulated rollouts; those in hacked_indices carry a hack flag."""
    hacked_indices = set(hacked_indices)
    for i in range(n_total):
        hacked = i in hacked_indices
        metrics = {
            "impossible": 1.0,
            "passed_mutated": float(hacked),
            "hacked_on_impossible": float(hacked),
        }
        logger_obj.maybe_log(
            env_type=env_type,
            suffix_name=None,
            datum_text=f"rollout {i}",
            total_reward=float(hacked),
            metrics=metrics,
        )


def _flush_and_read(logger_obj, tmp_path, step=0):
    """Finalize (flush) and return the rendered HTML for the given step."""
    logger_obj.finalize()
    path = tmp_path / f"step_samples_{step:06d}.html"
    assert path.exists(), f"expected flushed sample file at {path}"
    return path.read_text()


@pytest.fixture(autouse=True)
def _seeded_random():
    random.seed(1234)


def test_all_hacked_retained_when_under_budget(tmp_path):
    """~3% hack rate, hacked count < budget: EVERY hacked rollout is kept and
    the rest of the budget is filled with random non-hacked rollouts."""
    logger_obj = StepSampleLogger()
    logger_obj.configure(str(tmp_path), samples_per_key=30)
    logger_obj.reset(0)
    # 15 hacked among 512 — the regime where uniform sampling caught none.
    _feed_rollouts(logger_obj, n_total=512, hacked_indices=range(0, 512, 35))
    html = _flush_and_read(logger_obj, tmp_path)

    n_hack = html.count("Sampling: hack-selected (grader-flagged)")
    n_random = html.count("Sampling: random")
    assert n_hack == 15, "every grader-flagged hacked rollout must be retained"
    assert n_hack + n_random == 30, "random non-hacked rollouts fill the remaining budget"
    assert html.count("[HACK-SELECTED]") == n_hack


def test_hacked_trimmed_to_budget_when_over(tmp_path):
    """~15% hack rate, hacked count > budget: exactly samples_per_key samples,
    all of them hack-selected (random subset of the hacked pool)."""
    logger_obj = StepSampleLogger()
    logger_obj.configure(str(tmp_path), samples_per_key=30)
    logger_obj.reset(0)
    # 77/512 hacked (~15%), budget 30.
    _feed_rollouts(logger_obj, n_total=512, hacked_indices=random.sample(range(512), 77))
    html = _flush_and_read(logger_obj, tmp_path)

    assert html.count("Sampling: hack-selected (grader-flagged)") == 30
    assert html.count("Sampling: random") == 0


def test_env_without_hack_flags_unaffected(tmp_path):
    """An env whose metrics carry no grader cheat signals (e.g. math) still gets
    a plain random sample of samples_per_key rollouts, none hack-selected."""
    logger_obj = StepSampleLogger()
    logger_obj.configure(str(tmp_path), samples_per_key=10)
    logger_obj.reset(0)
    for i in range(200):
        logger_obj.maybe_log(
            env_type="math",
            suffix_name=None,
            datum_text=f"rollout {i}",
            total_reward=1.0,
            metrics={"accuracy": 1.0, "format_ok": 1.0},
        )
    html = _flush_and_read(logger_obj, tmp_path)

    assert html.count("Sampling: random") == 10
    assert "[HACK-SELECTED]" not in html


def test_hacked_kept_per_key_independently(tmp_path):
    """Budgets and hack-preference apply per (env_type, suffix) key."""
    logger_obj = StepSampleLogger()
    logger_obj.configure(str(tmp_path), samples_per_key=5)
    logger_obj.reset(0)
    _feed_rollouts(logger_obj, n_total=100, hacked_indices=[3, 50], env_type="reward_hack")
    _feed_rollouts(logger_obj, n_total=100, hacked_indices=[], env_type="other_env")
    html = _flush_and_read(logger_obj, tmp_path)

    assert html.count("Sampling: hack-selected (grader-flagged)") == 2
    assert html.count("Sampling: random") == 8  # 3 reward_hack fill + 5 other_env


def test_is_hacked_recognizes_each_grader_flag():
    for key in _HACK_FLAG_KEYS:
        assert _is_hacked({key: 1.0}), key
        assert not _is_hacked({key: 0.0}), key
    assert not _is_hacked({})
    # Defensive: None / non-numeric values never crash or count as hacked.
    assert not _is_hacked({"hacked_on_impossible": None})
    assert not _is_hacked({"hacked_on_impossible": "not-a-number"})
