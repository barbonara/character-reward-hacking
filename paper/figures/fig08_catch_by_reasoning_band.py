"""Post Fig 8: full-transcript monitor catch rate (mean of the three monitors at 2% FPR) for each hack-reasoning
band from the motivated-reasoning judge: rating 1 (fully genuine) .. 5 (fully motivated), plus the two silent
bands (hack never mentioned in the reasoning; with / without a comment on the override in the answer).
Every hacked held-out rollout with all three monitor scores and an MR judgment, every 10 RL steps pooled; the two
MR judges (Opus 5, DeepSeek V4.1 Flash) are pooled, one observation per rollout per judge; n = judgments;
95% Wilson intervals.
Needs $CORIN_MONITOR_DIR/judgments.jsonl and $CORIN_MR_DIR/judgments_{opus,deepseek}.jsonl."""
from paths import DATA_DIR, FIG_DIR, MR_DIR  # noqa: E402
import collections, json, os
import numpy as np, matplotlib.pyplot as plt
from orx_figstyle import COLUMN, TEXT, use_style, save
import monitor_common as MP
import mr_common as MR
use_style()
rows, _, _ = MP.load(); thr, _ = MP.thresholds(rows); per = MP.per_rollout(rows, thr)
JUDGES = ("haiku", "qwen", "deepseek")
ARMK = {"pro": "pro", "neutral": "neu", "anti": "anti"}
KEYS = ["r1", "r2", "r3", "r4", "r5", "silent", "silent_c"]
COL = json.load(open(os.path.join(DATA_DIR, "band_palette.json")))
LAB = {"r1": "1\nfully\ngenuine", "r2": "2", "r3": "3\nmixed", "r4": "4", "r5": "5\nfully\nmotivated",
       "silent": "hack not in\nreasoning;\nno comment", "silent_c": "hack not in\nreasoning;\nanswer comments\non the override"}


C = collections.defaultdict(list)     # rid -> [band per MR judge]
n_unparsed = 0                        # N judgments whose answer_claim can't be parsed: excluded
for jname in ("opus", "deepseek"):
    for line in open(os.path.join(MR_DIR, f"judgments_{jname}.jsonl")):
        o = json.loads(line)
        if o.get("hacked") and o.get("rating"):
            b = MR.band(o)
            if b is None:
                n_unparsed += 1
                continue
            rid = f"{ARMK[o['char']]}_s{o['seed']}/step{o['step']:02d}/{o['task_id']}/{o['id'].rsplit('/', 1)[1]}"
            C[rid].append(b)
if n_unparsed:
    print(f"excluded {n_unparsed} N judgments with an unparseable answer_claim")
caught = collections.defaultdict(list)
for rid, d in per["full"].items():
    if rid in C and all(j in d["flags"] for j in JUDGES):
        m = np.mean([d["flags"][j] for j in JUDGES])
        for k in C[rid]:
            caught[k].append(m)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - w), 100 * (c + w)


fig, ax = plt.subplots(figsize=(TEXT, COLUMN * 0.8), layout="constrained")
out = {}
for i, k in enumerate(KEYS):
    v = np.array(caught[k]); n = len(v)
    if n == 0:
        continue
    p = 100 * v.mean(); lo, hi = wilson(v.mean() * n, n)   # CI on pooled judgments (both MR judges)
    ax.bar(i, p, color=COL[k], width=0.62, linewidth=0)
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", linewidth=0.7, capsize=1.5)
    ax.text(i, hi + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.text(i, 1.5 if p > 8 else hi + 8, f"n={n}", ha="center", va="bottom", fontsize=5.5, color="white" if p > 8 else "black")
    out[k] = dict(n=n, catch=round(p, 1), lo=round(lo, 1), hi=round(hi, 1))
ax.set_xticks(range(len(KEYS))); ax.set_xticklabels([LAB[k] for k in KEYS], fontsize=6); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("Hacks caught by\nfull-transcript monitors (%)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
save(fig, os.path.join(FIG_DIR, "fig08_catch_by_reasoning_band"), formats=("png", "pdf"))
json.dump(out, open(os.path.join(FIG_DIR, "fig08_catch_by_reasoning_band.json"), "w"), indent=1)
for k in KEYS: print(k, out.get(k))
