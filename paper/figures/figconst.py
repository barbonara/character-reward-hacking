"""Shared constants and small helpers for the figure scripts."""
import os

import numpy as np
from orx_figstyle import BASELINE, MUTED, PALETTE
from paths import FIG_DIR

# One colour per character, fixed everywhere (Okabe-Ito); the untrained base is grey.
ARM_COLOR = {
    "pro": PALETTE["red"],
    "neu": PALETTE["blue"],
    "anti": PALETTE["green"],
    "base": MUTED,
    "van": BASELINE,
}
ARM_NAME = {
    "pro": "pro-cheating",
    "neu": "neutral",
    "anti": "anti-cheating",
    "base": "untrained base",
}
FORMATS = ("pdf", "svg", "png")


def out(stem):
    return os.path.join(FIG_DIR, stem)


def smooth(y, w=5):
    """Trailing w-step mean (shorter window at the start)."""
    y = np.asarray(y, float)
    return np.array([y[max(0, i - w + 1): i + 1].mean() for i in range(len(y))])


def wilson(h, n, z=1.96):
    """Wilson 95% interval for h successes out of n (h may be fractional)."""
    p = h / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - w, c + w


def cluster_boot_ci(values, clusters, n_boot=2000, seed=0, level=95):
    """Percentile CI for the pooled mean of `values`, resampling whole clusters (tasks) with
    replacement. Rollouts of the same held-out task (5 samples per checkpoint, many checkpoints)
    are not independent, so a CI that treats them as independent observations is too narrow.
    Returns (lo, hi) on the scale of `values`; (nan, nan) if empty."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan")
    keys, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(keys))
    cnts = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    a = (100 - level) / 2
    lo, hi = np.percentile(boots, [a, 100 - a])
    return float(lo), float(hi)


def need(path, how):
    """Open `path`, or exit with a hint on how to produce it (judge outputs are not shipped)."""
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {path}\n  {how}")
    return open(path)


MONITOR_HINT = ("produce it with `uv run python -m scripts.monitor_judge --judge <haiku|qwen|deepseek> "
                "--variants full,answer,ronly`, or point $CORIN_MONITOR_DIR at a directory holding judgments.jsonl")
MR_HINT = ("produce it with `uv run python -m scripts.mr_judge --phase hacked --judges opus,deepseek`, "
           "or point $CORIN_MR_DIR at a directory holding judgments_<judge>.jsonl")
