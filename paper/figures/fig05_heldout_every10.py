"""Post Fig 5: held-out impossible-task hack rate at every 10 RL steps, one line per run, no markers;
missing 10-step cells (the seed-3 runs at 10/20/30, neutral s3 at 0) are left as gaps."""
import json, os
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from paths import DATA_DIR, FIG_DIR, MR_PROMPT  # noqa: F401
from orx_figstyle import TEXT, use_style, save, figure_grid, panel_labels
from figconst import ARM_COLOR, ARM_NAME
use_style()
OUT = FIG_DIR
R = {k: v for k, v in json.load(open(os.path.join(DATA_DIR, "rl9_every10.json"))).items() if not k.startswith("_")}
ARMS = ["pro", "neu", "anti"]
def series(run):
    return [(p[0], 100 * p[1]) for p in R[run] if p[0] % 10 == 0]
def draw(ax, run, c, label=None):
    pts = series(run); ls = "-"
    # break the line where a 10-step cell is missing
    seg = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if b[0] - a[0] == 10: seg.append(b)
        else:
            ax.plot([p[0] for p in seg], [p[1] for p in seg], ls, color=c, lw=1.0); seg = [b]
    ax.plot([p[0] for p in seg], [p[1] for p in seg], ls, color=c, lw=1.0)
    if label: ax.text(pts[-1][0] + 1.2, pts[-1][1], label, fontsize=5.5, color=c, va="center")
fig, axes = figure_grid(2, 2, width=TEXT, ratio=0.72, sharex=True, sharey=True); axes = axes.ravel()
for ax, arm in zip(axes[:3], ARMS):
    c = ARM_COLOR[arm]; ys = []
    for s in (1, 2, 3):
        run = f"{arm}_s{s}"; pts = series(run); y = pts[-1][1]
        while any(abs(y - y0) < 5 for y0 in ys): y += 5
        ys.append(y); draw(ax, run, c)
        if arm == "anti": ax.text(pts[-1][0] + 1.2, y, f"s{s}", fontsize=5.5, color=c, va="center")
        draw(axes[3], run, c)
    ax.text(0.03, 0.96, ARM_NAME[arm], transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold", color=c)
axes[3].text(0.03, 0.96, "all characters", transform=axes[3].transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
for ax in axes:
    ax.set_xlim(-2, 97); ax.set_ylim(-4, 100); ax.set_xticks(range(0, 91, 10)); ax.set_yticks(range(0, 101, 20))
for ax in axes[2:]: ax.set_xlabel("RL step")
for ax in axes[::2]: ax.set_ylabel("Held-out impossible\ntasks hacked (%)")
h = [mpl.lines.Line2D([], [], color=ARM_COLOR[a], lw=1.0, label=ARM_NAME[a]) for a in ARMS]
fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False,
           title="every 10 steps, 110 held-out rollouts per point (anti s2 step 0: 86); one line per RL seed; all 90 cells present", title_fontsize=6.5)
panel_labels(axes)
save(fig, os.path.join(OUT, "fig05_heldout_every10"), formats=("png", "pdf", "svg"))
