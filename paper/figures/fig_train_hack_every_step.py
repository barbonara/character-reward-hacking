"""Supplementary (not in the post): training-rollout hack rate on impossible tasks, every step 0-89, unweighted trailing
5-step mean, one line per run, same layout as post Fig 5. The released early checkpoints sit where this curve
first reaches 50% (see README, "Released models")."""
import json, os
import numpy as np, matplotlib as mpl
from paths import DATA_DIR, FIG_DIR, MR_PROMPT  # noqa: F401
from orx_figstyle import TEXT, use_style, save, figure_grid, panel_labels
from figconst import ARM_COLOR, ARM_NAME, smooth
use_style()
OUT = FIG_DIR
R = json.load(open(os.path.join(DATA_DIR, "rl9.json")))["train_hack"]
NS = [p[2] for v in R.values() for p in v]  # impossible-task training rollouts per step
ARMS = ["pro", "neu", "anti"]
def series(run):
    pts = np.array(R[run]); return pts[:, 0], 100 * smooth(pts[:, 1], 5)
fig, axes = figure_grid(2, 2, width=TEXT, ratio=0.72, sharex=True, sharey=True); axes = axes.ravel()
for ax, arm in zip(axes[:3], ARMS):
    c = ARM_COLOR[arm]; ys = []
    for s in (1, 2, 3):
        x, y = series(f"{arm}_s{s}")
        ax.plot(x, y, "-", color=c, lw=1.0); axes[3].plot(x, y, "-", color=c, lw=1.0)
        if arm == "anti":
            yy = y[-1]
            while any(abs(yy - y0) < 5 for y0 in ys): yy += 5
            ys.append(yy); ax.text(x[-1] + 1.2, yy, f"s{s}", fontsize=5.5, color=c, va="center")
    ax.text(0.03, 0.96, ARM_NAME[arm], transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold", color=c)
axes[3].text(0.03, 0.96, "all characters", transform=axes[3].transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
for ax in axes:
    ax.set_xlim(-2, 97); ax.set_ylim(-4, 100); ax.set_xticks(range(0, 91, 10)); ax.set_yticks(range(0, 101, 20))
for ax in axes[2:]: ax.set_xlabel("RL step")
for ax in axes[::2]: ax.set_ylabel("Training impossible-task\nrollouts hacked (%)")
h = [mpl.lines.Line2D([], [], color=ARM_COLOR[a], lw=1.0, label=ARM_NAME[a]) for a in ARMS]
fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=6.5, frameon=False,
           title=f"every training step, trailing 5-step mean of the batch hack rate ({min(NS)}–{max(NS)} impossible-task rollouts per step); one line per RL seed", title_fontsize=6.5)
panel_labels(axes)
save(fig, os.path.join(OUT, "figS1_train_hack_every_step"), formats=("png", "pdf", "svg"))
