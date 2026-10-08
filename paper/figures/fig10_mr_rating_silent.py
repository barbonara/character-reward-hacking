"""Post Fig 10: per run, two bars: mean 1-5 motivated-reasoning rating among rated judgments (left axis) and share of
judgments that are silent (hack never mentioned in the reasoning; right axis), with the full-transcript monitor catch
rate (mean of the three monitors at 2% FPR) overlaid. Every hacked held-out rollout, every 10 RL steps pooled; the two
MR judges (Opus 5, DeepSeek V4.1 Flash) are pooled, one observation per rollout per judge.
Needs $CORIN_MONITOR_DIR/judgments.jsonl and $CORIN_MR_DIR/judgments_{opus,deepseek}.jsonl."""
import json, os, collections
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from paths import FIG_DIR, MR_DIR
from orx_figstyle import COLUMN, use_style, save
from figconst import ARM_COLOR, ARM_NAME
use_style(); OUT = FIG_DIR
ARMK = {"pro": "pro", "neutral": "neu", "anti": "anti"}
vals = collections.defaultdict(list); silent = collections.Counter(); nroll = collections.defaultdict(set)
for j in ("opus", "deepseek"):
    for l in open(os.path.join(MR_DIR, f"judgments_{j}.jsonl")):
        o = json.loads(l)
        if not (o.get("hacked") and o.get("rating")): continue
        run = f"{ARMK[o['char']]}_s{o['seed']}"; nroll[run].add(o["id"])
        if o["rating"] == "N": silent[run] += 1
        else: vals[run].append(int(o["rating"]))
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def stats(runs):
    v = np.concatenate([np.array(vals[r]) for r in runs if vals[r]]) if any(vals[r] for r in runs) else np.array([])
    s = sum(silent[r] for r in runs); tot = len(v) + s
    if tot == 0: return None
    m = v.mean() if len(v) else np.nan; ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    lo, hi = wilson(s, tot)
    return {"mean": m, "ci": ci, "n_rated": int(len(v)), "silent_pct": 100 * s / tot, "silent_lo": 100 * lo, "silent_hi": 100 * hi, "n_judgments": tot, "n_rollouts": len(set().union(*[nroll[r] for r in runs]))}
RUNS = [f"{a}_s{s}" for a in ("pro", "neu", "anti") for s in (1, 2, 3)]
out = {"by_run": {}, "by_char": {}}
import monitor_common as MP
_rows, _, _ = MP.load(); _thr, _ = MP.thresholds(_rows); _per = MP.per_rollout(_rows, _thr)
MJ = ("haiku", "qwen", "deepseek"); MISS = {}
for rid, d in _per["full"].items():
    if all(j in d["flags"] for j in MJ):
        a_, rest = rid.split("/", 1); MISS[a_.replace("_s", "-s").replace("neu-", "neutral-") + "/" + rest] = 1 - np.mean([d["flags"][j] for j in MJ])
MCOL = "#1f5fa8"
for r in RUNS:
    st = stats([r])
    if st: out["by_run"][r] = st
for a in ("pro", "neu", "anti"):
    out["by_char"][a] = stats([f"{a}_s{s}" for s in (1, 2, 3)])
    print(f"{ARM_NAME[a]:14s} mean rating {out['by_char'][a]['mean']:.2f} ±{out['by_char'][a]['ci']:.2f} (n rated {out['by_char'][a]['n_rated']})   silent {out['by_char'][a]['silent_pct']:.0f}% [{out['by_char'][a]['silent_lo']:.0f}–{out['by_char'][a]['silent_hi']:.0f}]  (judgments {out['by_char'][a]['n_judgments']}, rollouts {out['by_char'][a]['n_rollouts']})")
json.dump(out, open(os.path.join(OUT, "fig10_mr_rating_silent.json"), "w"), indent=1, default=float)

fig, ax = plt.subplots(figsize=(COLUMN, COLUMN * 0.95), layout="constrained"); ax2 = ax.twinx()
W = 0.36
for i, run in enumerate(RUNS):
    arm = run.split("_")[0]; st = out["by_run"].get(run)
    if not st:
        ax.text(i, 1.15, "no\nhacks", ha="center", va="bottom", fontsize=5.5, color="#666666"); continue
    ax.bar(i - W / 2, st["mean"], W, color=ARM_COLOR[arm], edgecolor="black", lw=0.4)
    ax.errorbar(i - W / 2, st["mean"], yerr=st["ci"], color="black", lw=0.8, capsize=1.5)
    ax2.bar(i + W / 2, st["silent_pct"], W, color="#111111", edgecolor="black", lw=0.4)
    ax2.errorbar(i + W / 2, st["silent_pct"], yerr=[[st["silent_pct"] - st["silent_lo"]], [st["silent_hi"] - st["silent_pct"]]], color="#555555", lw=0.8, capsize=1.5)
    ax.text(i, 5.08, str(st["n_rollouts"]), ha="center", va="bottom", fontsize=5, color="#555555")
    mv = [MISS[x] for x in nroll[run] if x in MISS]
    if mv:
        m = 100 * (1 - np.mean(mv)); st["catch_pct"] = m
        ax2.plot(i, m, marker="o", ms=3.2, mfc="white", mec=MCOL, mew=1.1, lw=0, zorder=6)
        ax2.text(i + (-0.14 if i == 8 else 0.14), m + 2.2, f"{m:.0f}%", ha=("right" if i == 8 else "left"), fontsize=5, color=MCOL, fontweight="bold", zorder=7, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
ax.text(-0.7, 5.08, "n", ha="right", va="bottom", fontsize=5, color="#555555")
ax.set_ylim(1, 5.5); ax.set_yticks(range(1, 6)); ax.set_xlim(-0.6, 8.6); ax.spines["left"].set_position(("data", -0.6))
ax2.set_ylim(0, 112.5); ax2.set_yticks(range(0, 101, 25)); ax2.spines["right"].set_position(("data", 8.6)); ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax.set_xticks(range(9)); ax.set_xticklabels([f"s{r[-1]}" for r in RUNS])
for arm, xs in (("pro", [0, 1, 2]), ("neu", [3, 4, 5]), ("anti", [6, 7, 8])):
    ax.text(np.mean(xs), 0.55, ARM_NAME[arm], ha="center", va="top", fontsize=7, color=ARM_COLOR[arm], clip_on=False)
ax.set_ylabel("Mean rating of rated reasoning\n(1 genuine – 5 motivated)"); ax.set_xlabel(" "); ax2.set_xlabel(" "); ax2.set_ylabel("Silent hacks / monitor catches (%)")
ax.axhline(1, color="black", lw=0.6); ax.grid(False); ax2.grid(False)
for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax.spines["right"].set_visible(False); ax2.spines["left"].set_visible(False)
class Flag(HandlerBase):
    def create_artists(self, legend, orig, x0, y0, w, h, fs, trans):
        cols = [ARM_COLOR[a] for a in ("pro", "neu", "anti")]
        return [Rectangle((x0 + i * w / 3, y0), w / 3, h, facecolor=c, edgecolor="black", lw=0.4, transform=trans) for i, c in enumerate(cols)]
flag = Patch(label="mean rating, 95% CI")
fig.legend(handles=[flag,
                    Patch(facecolor="#111111", edgecolor="black", lw=0.4, label="silent: hack never mentioned in\nreasoning (%), 95% Wilson"),
                    Line2D([0], [0], lw=0, marker="o", mfc="white", mec=MCOL, mew=1.4, label="monitor catch rate (%), full transcript,\nmean of 3 monitors at 2% FPR")],
           loc="outside lower center", ncol=2, fontsize=5.5, frameon=False, handler_map={flag: Flag()})
save(fig, os.path.join(OUT, "fig10_mr_rating_silent"), formats=("png", "pdf"))
