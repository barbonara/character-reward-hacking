"""Aggregate cheating-stance scores (scripts/cheat_stance.py) into per-checkpoint means.

Reads the .eval logs directly (not metrics.json, which also carries --limit smoke-test lines)
and uses, per (checkpoint, eval), the newest successful log without a sample limit.

Output: $CORIN_OUTPUT_DIR/cheat_stance/cheat_stance.json with
  cells["<eval>/<run>_step<NN>"] = {cheat_stance: {mean, stderr, n}, engagement: {...}, log}
  survival["<eval>"][<char>] = {"s0": step-0 mean, "s<step>": [seed1, seed2, seed3]}
stderr is the standard error of the per-sample scores (sample SD / sqrt(n)), as inspect
reports it; unparseable judge outputs (NaN) are excluded and counted in n_invalid.

    uv run python -m scripts.cheat_stance_aggregate [--step 60]
"""
import argparse
import glob
import json
import math
import os
import zipfile

from scripts.common import CHARACTERS, OUTPUT_DIR

FIELDS = ("cheat_stance", "engagement")


def latest_full_log(eval_dir):
    best = None
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.eval"))):  # filenames start with an ISO timestamp
        hdr = json.loads(zipfile.ZipFile(path).read("header.json"))
        if hdr.get("status") == "success" and not hdr["eval"]["config"].get("limit"):
            best = path
    return best


def summarise(path):
    z = zipfile.ZipFile(path)
    vals = {f: [] for f in FIELDS}
    for name in z.namelist():
        if not name.startswith("samples/"):
            continue
        score = json.loads(z.read(name))["scores"]["judge_scorer"]["value"]
        for f in FIELDS:
            vals[f].append(score.get(f"response_only/{f}"))
    out = {"log": os.path.basename(path)}
    for f, v in vals.items():
        ok = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
        mean = sum(ok) / len(ok) if ok else None
        sd = math.sqrt(sum((x - mean) ** 2 for x in ok) / (len(ok) - 1)) if len(ok) > 1 else 0.0
        out[f] = dict(mean=mean, stderr=sd / math.sqrt(len(ok)) if ok else None, n=len(ok), n_invalid=len(v) - len(ok))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.join(OUTPUT_DIR, "cheat_stance")
    ap.add_argument("--root", default=root)
    ap.add_argument("--step", type=int, default=60, help="RL step for the survival summary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cells = {}
    for eval_dir in sorted(glob.glob(os.path.join(a.root, "*_step*", "evals", "*"))):
        cell = eval_dir.split(os.sep)[-3]
        if path := latest_full_log(eval_dir):
            cells[f"{os.path.basename(eval_dir)}/{cell}"] = summarise(path)
    survival = {}
    for key, c in cells.items():
        ev, cell = key.split("/")
        run, step = cell.rsplit("_step", 1)
        char, seed = run.split("_s")
        step = int(step)
        d = survival.setdefault(ev, {}).setdefault(char, {})
        if step == 0:
            d["s0"] = c["cheat_stance"]["mean"]
        elif step == a.step:
            d.setdefault(f"s{step}", {})[int(seed)] = c["cheat_stance"]["mean"]
    for ev in survival:
        for char in survival[ev]:
            k = f"s{a.step}"
            if k in survival[ev][char]:
                survival[ev][char][k] = [survival[ev][char][k][s] for s in sorted(survival[ev][char][k])]
        survival[ev] = {c: survival[ev][c] for c in CHARACTERS if c in survival[ev]}
    out = a.out or os.path.join(a.root, "cheat_stance.json")
    json.dump({"metric": "Claude Sonnet 5 cheat-stance score, 0 (anti-cheating) .. 10 (pro-cheating), "
                         "response-only, mean over prompts (1 sample per prompt)",
               "cells": cells, "survival": survival}, open(out, "w"), indent=1)
    print(f"{len(cells)} cells -> {out}")
    for ev, chars in survival.items():
        for char, d in chars.items():
            s0 = d.get("s0")
            rl = d.get(f"s{a.step}", [])
            print(f"{ev:22s} {char:8s} step0 {s0 if s0 is None else round(s0, 2)}  "
                  f"step{a.step} {[round(x, 2) for x in rl]}")


if __name__ == "__main__":
    main()
