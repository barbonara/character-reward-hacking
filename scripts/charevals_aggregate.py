"""Aggregate cross-spec character-expression scores into per-cell means (read by paper/figures/fig11_expression_crossspec.py and fig12_expression_baselines.py).

Input:  $CORIN_OUTPUT_DIR/charevals/crossspec_scores.jsonl (scripts/charevals_judge.py)
Output: $CORIN_OUTPUT_DIR/charevals/charevals_crossspec_step0.json, the same layout as the shipped
        paper/figures/data/charevals_crossspec_step0.json: {metric, judge, cells_key, cells} with
        cells["<eval>/<arm>/<spec>"] = {mean, n, n_invalid}, scores normalised 0-1. By default only the
        step-0 and untrained-base cells (what the figures read); --all-cells adds the RL cells.
Arm keys: step-0 (SFT) cells are the character ("pro" / "neu" / "anti", pooled over the
seed directories it was sampled under); RL cells are "<char>_s<seed>_s<step>", e.g. "neu_s2_s90";
the untrained base is "base" ("You are Corin.") and "base_nemotron" ("You are Nemotron.").

    uv run python -m scripts.charevals_aggregate
"""
import argparse
import collections
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = os.path.join(OUTPUT_DIR, "charevals")
    ap.add_argument("--scores", default=os.path.join(d, "crossspec_scores.jsonl"))
    ap.add_argument("--out", default=os.path.join(d, "charevals_crossspec_step0.json"))
    ap.add_argument("--all-cells", action="store_true", help="also aggregate RL-checkpoint cells (<char>_s<seed>_s<step>)")
    a = ap.parse_args()
    # First record per (eval, model, spec, sample) wins; a later record replaces it only if the
    # first one has no valid score (a re-judged unparseable output).
    best = {}
    for line in open(a.scores):
        r = json.loads(line)
        k = (r["eval"], r["model"], r["spec"], r["sample_id"])
        if k not in best or (best[k]["score"] is None and r["score"] is not None):
            best[k] = r
    cells = collections.defaultdict(list)
    for r in best.values():
        arm = arm_key(r["model"])
        if a.all_cells or "_s" not in arm:  # RL cells are "<char>_s<seed>_s<step>"
            cells[f"{r['eval']}/{arm}/{r['spec']}"].append(r["score"])
    out = {}
    for k, scores in sorted(cells.items()):
        v = [s / 10 for s in scores if s is not None]
        out[k] = dict(mean=sum(v) / len(v) if v else None, n=len(v), n_invalid=len(scores) - len(v))
    json.dump({"metric": "mean judge score (integer 0-10 per sample) normalised to 0-1; "
                         "n_invalid = unparseable judge output (excluded)",
               "judge": "anthropic/claude-sonnet-5 via OpenRouter, response-only, reasoning disabled, temperature unset, "
                        "max_tokens 1024 (scripts/charevals_judge.py)",
               "cells_key": "<eval>/<arm>/<spec>; arm = step-0 SFT checkpoint (pro / neu / anti), base = untrained base "
                            'under "You are Corin.", base_nemotron = untrained base under "You are Nemotron."'
                            + ("; RL cells <char>_s<seed>_s<step>" if a.all_cells else ""),
               "cells": out}, open(a.out, "w"), indent=1)
    print(f"{len(out)} cells -> {a.out}")


if __name__ == "__main__":
    main()
