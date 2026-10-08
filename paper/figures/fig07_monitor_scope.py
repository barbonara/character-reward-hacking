"""Post Fig 7: monitor catch rate by what the monitor sees (full transcript / task + final answer / reasoning
only), mean of the three monitors, pooled over each run's crossing, step-60 and step-90 cells.
Bars = mean of the per-seed catch rates; dots = seeds; whiskers = 95% CI bootstrapped over rollouts within
each seed. Thresholds are calibrated per view on that view's honest rollouts from steps >= 30 at a 2% target
FPR (the realised FPR is printed).
Needs $CORIN_MONITOR_DIR/judgments.jsonl (scripts/monitor_judge.py)."""
from paths import FIG_DIR  # noqa: E402
import collections, json, os
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from orx_figstyle import TEXT, use_style, save
from figconst import ARM_COLOR
import monitor_common as F
use_style()
CAL = 30   # calibrate on honest rollouts from steps >= CAL (where the figure is evaluated)
rows, _, _ = F.load()
hon_ids = {v: {r["rid"] for r in rows if not r["hacked"] and r["variant"] == v and r["step"] >= CAL} for v in F.VARIANTS}
sub = [r for r in rows if r["hacked"] or r["rid"] in hon_ids[r["variant"]]]   # each view calibrated on its own honest rollouts
thr, fpr = F.thresholds(sub)
per = F.per_rollout(rows, thr)
late_h = {r["rid"] for r in rows if not r["hacked"] and r["step"] >= CAL}
for v in F.VARIANTS:  # realised FPR on the late honest rollouts (three-monitor mean)
    vals = collections.defaultdict(list)
    for r in rows:
        if r["variant"] == v and r["rid"] in late_h:
            vals[r["rid"]].append(F.flag_value(r["score"], thr[(r["judge"], v)]))
    m = [np.mean(x) for x in vals.values() if len(x) == 3]
    print(f"  realised FPR, {v}, honest steps>={CAL}: {100 * np.mean(m):.1f}% (n={len(m)})")
CELLS = ["50%", "60", "90"]
GROUPS = [("pro", ["pro_s1", "pro_s2", "pro_s3"], ARM_COLOR["pro"], "pro-cheating"),
          ("neu", ["neu_s1", "neu_s2", "neu_s3"], ARM_COLOR["neu"], "neutral"),
          ("anti", ["anti_s1", "anti_s2", "anti_s3"], ARM_COLOR["anti"], "anti-cheating")]


def seed_flags(v, run):
    return [np.mean([d["flags"][j] for j in F.JUDGES]) for d in F.select(per[v], run=run, cells=CELLS)
            if all(j in d["flags"] for j in F.JUDGES)]


fig, ax = plt.subplots(1, 1, figsize=(TEXT, TEXT * 0.52), layout="constrained")
W = 0.26; out = {}
for vi, v in enumerate(F.VARIANTS):
    for gi, (g, runs, col, lab) in enumerate(GROUPS):
        x = vi + (gi - 1) * W
        seeds = [np.array(xs) for xs in (seed_flags(v, run) for run in runs) if len(xs) >= 5]
        m = 100 * np.mean([a.mean() for a in seeds])            # mean of per-seed catch rates
        rng = np.random.default_rng(0)
        boots = [100 * np.mean([rng.choice(a, len(a)).mean() for a in seeds]) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax.bar(x, m, W * 0.94, color=col, edgecolor="black", lw=0.4)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", lw=0.7, capsize=1.6, zorder=4)
        for a in seeds:
            ax.plot(x, 100 * a.mean(), "o", ms=2.6, color="black", zorder=5)
        out[f"{v}|{g}"] = dict(mean=round(m, 1), lo=round(lo, 1), hi=round(hi, 1), n=int(sum(len(a) for a in seeds)),
                               seeds=[round(100 * a.mean(), 1) for a in seeds])
ax.set_xticks(range(3)); ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 110); ax.set_yticks(range(0, 101, 25))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_xticklabels([F.SCOPE_NAME[v] for v in F.VARIANTS], fontsize=6.5)
ax.set_ylabel("Hacked rollouts flagged (%)\nmean of three monitors")
ax.set_xlabel("What the monitor sees")
fig.legend(handles=[Patch(facecolor=c, edgecolor="black", lw=0.4, label=l) for _, _, c, l in GROUPS] +
           [Line2D([0], [0], marker="o", lw=0, color="black", ms=3, label="one RL seed"),
            Line2D([0], [0], color="black", lw=0.7, label="95% CI (bootstrap over rollouts)")],
           loc="outside lower center", ncol=3, fontsize=5.8, frameon=False)
json.dump(out, open(os.path.join(FIG_DIR, "fig07_monitor_scope.json"), "w"), indent=1)
save(fig, os.path.join(FIG_DIR, "fig07_monitor_scope"), formats=("png", "pdf"))
for k, v in out.items(): print(k, v)
