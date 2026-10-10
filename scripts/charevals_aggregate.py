"""Aggregate cross-spec character-expression scores into per-cell means (read by paper/figures/fig11_expression_crossspec.py and fig12_expression_baselines.py).

Input:  $CORIN_OUTPUT_DIR/charevals/crossspec_scores.jsonl (scripts/charevals_judge.py)
Output: $CORIN_OUTPUT_DIR/charevals/charevals_crossspec_step0.json, the file the figures read
        (shipped as paper/figures/data/charevals_crossspec_step0.json), with
        cells["<eval>/<arm>/<spec>"] = {mean, n, n_invalid, ci95}, scores normalised 0-1.
Arm keys: step-0 (SFT) cells are the character ("pro" / "neu" / "anti", pooled over the
seed directories it was sampled under); the untrained base is "base" ("You are Corin.") and
"base_nemotron" ("You are Nemotron."). With --include-rl, RL cells ("<char>_s<seed>_s<step>",
e.g. "neu_s2_s90") are kept too and the default output is charevals_crossspec.json.

To draw figs 11-12 from your own aggregate, point $CORIN_FIG_DATA at $CORIN_OUTPUT_DIR/charevals
(those two figures read only this file); the shipped data file is never overwritten by default.

    uv run python -m scripts.charevals_aggregate
"""
import argparse
import collections
import json
import math
import os

from scripts.common import OUTPUT_DIR

SHORT = {"pro": "pro", "neutral": "neu", "anti": "anti"}


def arm_key(model):
    run, step = model.rsplit("_step", 1)
    if run == "base_corin":  # untrained base under "You are Corin."
        return "base"
    if run == "base_nemotron":  # untrained base under "You are Nemotron."
        return "base_nemotron"
    char, seed = run.split("_s")
    return SHORT[char] if int(step) == 0 else f"{SHORT[char]}_s{seed}_s{int(step)}"


STEP0_ARMS = {"pro", "neu", "anti", "base", "base_nemotron"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = os.path.join(OUTPUT_DIR, "charevals")
    ap.add_argument("--scores", default=os.path.join(d, "crossspec_scores.jsonl"))
    ap.add_argument("--include-rl", action="store_true", help="keep RL-checkpoint cells as well")
    ap.add_argument("--out", default=None, help="default: charevals_crossspec_step0.json "
                                                "(charevals_crossspec.json with --include-rl) next to --scores")
    a = ap.parse_args()
    out_path = a.out or os.path.join(os.path.dirname(a.scores),
                                     "charevals_crossspec.json" if a.include_rl else "charevals_crossspec_step0.json")
    # First valid record per (eval, model, spec, sample), else the first record. charevals_judge.py
    # appends and skips keys that already have a valid score, so the first valid one is the one it kept.
    best, judges = {}, set()
    for line in open(a.scores):
        r = json.loads(line)
        k = (r["eval"], r["model"], r["spec"], r["sample_id"])
        if k not in best or (best[k]["score"] is None and r["score"] is not None):
            best[k] = r
    cells = collections.defaultdict(list)
    for r in best.values():
        arm = arm_key(r["model"])
        if a.include_rl or arm in STEP0_ARMS:
            cells[f"{r['eval']}/{arm}/{r['spec']}"].append(r["score"])
            judges.add(r.get("judge"))
    if len(judges) > 1:
        print(f"WARNING: scores from more than one judge are pooled: {sorted(map(str, judges))}")
    out = {}
    for k, scores in sorted(cells.items()):
        v = [s / 10 for s in scores if s is not None]
        mean = sum(v) / len(v) if v else None
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        out[k] = dict(mean=mean, n=len(v), n_invalid=len(scores) - len(v), ci95=1.96 * sd / math.sqrt(len(v)) if v else 0.0)
    json.dump({"metric": "mean judge score (integer 0-10 per sample) normalised to 0-1; "
                         "n_invalid = unparseable judge output (excluded)",
               "judge": ", ".join(sorted(map(str, judges))) + " (scripts/charevals_judge.py)",
               "cells_key": "<eval>/<arm>/<spec>; see scripts/charevals_aggregate.py for the arm keys",
               "cells": out}, open(out_path, "w"), indent=1)
    print(f"{len(out)} cells -> {out_path}")


if __name__ == "__main__":
    main()
