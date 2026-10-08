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
