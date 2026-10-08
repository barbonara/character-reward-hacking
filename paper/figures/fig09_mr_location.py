"""Post Fig 9: where the hack shows up in the transcript, per run. Top: all judged checkpoints pooled; bottom: per
checkpoint every 10 RL steps, one panel per run; full-transcript monitor catch rate overlaid. MR judges pooled
(Opus 5 + DeepSeek V4.1 Flash, one observation per rollout per judge).
Needs $CORIN_MONITOR_DIR/judgments.jsonl and $CORIN_MR_DIR/judgments_{opus,deepseek}.jsonl."""
import json, os, collections
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from paths import DATA_DIR, FIG_DIR, MR_DIR
from orx_figstyle import TEXT, use_style
from orx_figstyle import allow_unlabeled, save
from figconst import ARM_COLOR, ARM_NAME
use_style(); OUT = FIG_DIR
ARMK = {"pro": "pro", "neutral": "neu", "anti": "anti"}; STEPS = list(range(0, 91, 10))
KEYS = ["r1", "r2", "r3", "r4", "r5", "silent", "silent_c"]
COL = json.load(open(os.path.join(DATA_DIR, "band_palette.json")))
LAB = {"r1": "rating 1 (fully genuine, openly gaming the test)", "r2": "rating 2 (mostly genuine)", "r3": "rating 3 (mixed)",
       "r4": "rating 4 (mostly rationalised)", "r5": "rating 5 (fully motivated)",
       "silent": "hack not mentioned in reasoning; no comment in answer", "silent_c": "hack not mentioned in reasoning; answer comments on the override"}
def cat(r, c):
    if r == "N": return "silent" if c == "none" else "silent_c"
    return "r" + r
cells = collections.defaultdict(collections.Counter); ncell = collections.defaultdict(set)
tot = collections.defaultdict(collections.Counter); nrun = collections.defaultdict(set)
for j in ("opus", "deepseek"):
    for l in open(os.path.join(MR_DIR, f"judgments_{j}.jsonl")):
        o = json.loads(l)
        if o.get("hacked") and o.get("rating"):
            arm = ARMK[o["char"]]; k = cat(o["rating"], o.get("answer_claim"))
            cells[(arm, o["seed"], o["step"])][k] += 1; ncell[(arm, o["seed"], o["step"])].add(o["id"])
            tot[f"{arm}_s{o['seed']}"][k] += 1; nrun[f"{arm}_s{o['seed']}"].add(o["id"])
def alpha_n(n): return max(0.15, min(1.0, n / 30))
import monitor_common as MP
_rows, _, _ = MP.load(); _thr, _ = MP.thresholds(_rows); _per = MP.per_rollout(_rows, _thr)
MJ = ("haiku", "qwen", "deepseek"); MISS = {}
for rid, d in _per["full"].items():
    if all(j in d["flags"] for j in MJ):
        a, rest = rid.split("/", 1); MISS[a.replace("_s", "-s").replace("neu-", "neutral-") + "/" + rest] = 1 - np.mean([d["flags"][j] for j in MJ])
def miss_rate(ids):
    v = [MISS[i] for i in ids if i in MISS]; return (100 * (1 - np.mean(v)), len(v)) if v else (None, 0)
MCOL = "#1f5fa8"

fig = plt.figure(figsize=(TEXT, TEXT * 1.42), layout="constrained")
gs = fig.add_gridspec(4, 3, height_ratios=[1.35, 1, 1, 1], hspace=0.08, wspace=0.06)

# ---- top: pooled per-run bars ----
ax = fig.add_subplot(gs[0, :])
RUNS = [f"{a}_s{s}" for a in ("pro", "neu", "anti") for s in (1, 2, 3)]
TOPM = {}
for i, run in enumerate(RUNS):
    c = tot.get(run, collections.Counter()); n = sum(c.values())
    if n == 0:
        ax.text(i, 3, "no hacks in\njudged steps", ha="center", va="bottom", fontsize=6, color="#666666"); continue
    b = 0
    for k in KEYS:
        v = 100 * c[k] / n
        if v: ax.bar(i, v, bottom=b, color=COL[k], width=0.72, linewidth=0.5, edgecolor="black"); b += v
    ax.text(i, 101.5, f"n={len(nrun[run])}", ha="center", va="bottom", fontsize=5.6, color="#444444")
    m, _ = miss_rate(nrun[run]); TOPM[i] = m
for grp in ([0, 1, 2], [3, 4, 5], [6, 7, 8]):
    xs = [i for i in grp if TOPM.get(i) is not None]
    seg = []
    for i in grp + [None]:
        if i is not None and TOPM.get(i) is not None: seg.append(i); continue
        if seg: ax.plot(seg, [TOPM[j] for j in seg], color=MCOL, lw=0, marker="o", ms=4.5, mfc="white", mec=MCOL, mew=1.4, zorder=5)
        seg = []
    for i in xs: ax.text(i + 0.12, TOPM[i] + (-7 if TOPM[i] > 88 else 2), f"{TOPM[i]:.0f}%", fontsize=5.8, color=MCOL, fontweight="bold", zorder=6,
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
ax.set_xticks(range(len(RUNS))); ax.set_xticklabels([f"s{r[-1]}" for r in RUNS], fontsize=6.5)
for arm, xs in (("pro", [0, 1, 2]), ("neu", [3, 4, 5]), ("anti", [6, 7, 8])):
    ax.text(np.mean(xs), -13, ARM_NAME[arm], ha="center", va="top", fontsize=7, color=ARM_COLOR[arm], clip_on=False)
ax.set_xlim(-0.6, 8.6); ax.set_ylim(0, 110); ax.set_xlabel(" "); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Hacked held-out rollouts (%)")
ax.set_title("All checkpoints pooled (steps 0–90)", loc="left", fontsize=7.5, fontweight="bold", pad=3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ---- bottom: 3x3 over steps ----
axes = np.empty((3, 3), dtype=object)
for ai, arm in enumerate(("pro", "neu", "anti")):
    for si, s in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[ai + 1, si], sharey=axes[0, 0] if (ai or si) else None); axes[ai, si] = ax
        for i, st in enumerate(STEPS):
            c = cells.get((arm, s, st), collections.Counter()); n = sum(c.values()); nr = len(ncell.get((arm, s, st), ()))
            if n == 0: continue
            b = 0; al = alpha_n(nr)
            for k in KEYS:
                v = 100 * c[k] / n
                if v: ax.bar(i, v, bottom=b, color=COL[k], width=0.8, linewidth=0.4, edgecolor="black", alpha=al); b += v
            ax.text(i, 101.5, str(nr), ha="center", va="bottom", fontsize=5, color="#444444")
        pts = [(i, miss_rate(ncell.get((arm, s, st), ()))[0]) for i, st in enumerate(STEPS)]
        pts = [(i, m) for i, m in pts if m is not None]
        if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MCOL, lw=1.4, marker="o", ms=3.2, mfc="white", mec=MCOL, mew=1.1, zorder=5)
        ax.set_xticks(range(len(STEPS))); ax.set_xticklabels(STEPS, fontsize=5.5); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
        ax.set_title(f"{ARM_NAME[arm]} · seed {s}", loc="left", fontsize=7, fontweight="bold", pad=2)
        if ai == 2: ax.set_xlabel("RL step")
        else: allow_unlabeled(ax, "x")  # inner rows: "RL step" is on the bottom row only
        if si == 0: ax.set_ylabel("Hacked rollouts (%)")
        else: ax.tick_params(labelleft=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if arm == "anti" and s == 2: ax.text(0.5, 0.5, "no hacks in the judged steps\n(12 of 11,520 training rollouts\nhacked across 90 steps)", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#666666")

from matplotlib.lines import Line2D
fig.legend([Patch(facecolor=COL[k], edgecolor="black", linewidth=0.4) for k in KEYS] + [Line2D([0], [0], color=MCOL, lw=1.6, marker="o", mfc="white", mec=MCOL)],
           [LAB[k] for k in KEYS] + ["monitor catch rate (full transcript; mean of 3 monitors at 2% FPR)"],
           loc="outside lower center", ncol=2, fontsize=5.8, frameon=False)

cap = ("Where the hack shows up in the transcript, per run. Top: all judged checkpoints pooled "
       "(steps 0–90), one stacked bar per run; n = distinct hacked held-out rollouts. Bottom: the same bands per checkpoint every 10 RL steps, "
       "one panel per run; the number above each bar is the count of hacked rollouts in that cell and bar opacity is proportional to it "
       "(full at 30). Bands are the motivated-reasoning judge's rating of the reasoning text (1 = fully genuine, including openly gaming the "
       "test; 5 = fully motivated), plus two silent bands for rollouts whose answer alters the test while the reasoning never mentions the "
       "edit, split by whether the answer carries a comment on the override. Judges: Opus 5 and DeepSeek V4.1 Flash, pooled as one "
       "observation per rollout per judge. Anti-cheating seed 2 produced no hacks in any judged held-out cell (steps 0, 10, …, 90); it did hack rarely elsewhere: 3 of 110 held-out rollouts at step 2, and 12 of 11,520 impossible-side training rollouts across the 90 steps.")
save(fig, os.path.join(OUT, "fig09_mr_location"), formats=("png", "pdf"))
with open(os.path.join(OUT, "fig09_mr_location.caption.txt"), "w") as f:  # overwritten on every run
    f.write(cap + "\n")
print("ok")
