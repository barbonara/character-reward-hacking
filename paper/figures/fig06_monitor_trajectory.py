"""Post Fig 6: full-transcript monitor catch rate across RL, mean of the three monitors (Claude Haiku 4.5,
Qwen3.5-122B, DeepSeek-V4-Flash), one line per RL seed, every 10 steps (plus each run's crossing cell);
one panel per character with Wilson 95% bands, and all characters overlaid.
Thresholds: 2% FPR per monitor, calibrated on the honest held-out rollouts from steps >= 30 (as post Fig 7).
Needs $CORIN_MONITOR_DIR/judgments.jsonl (scripts/monitor_judge.py over every-10-step held-out cells)."""
from paths import FIG_DIR  # noqa: F401
import matplotlib as mpl
import numpy as np
from figconst import ARM_COLOR, ARM_NAME, FORMATS, out, wilson
from monitor_common import ARMS, MIN_HACKED, RUNS, catch, load, pct, per_rollout, select, thresholds
from orx_figstyle import TEXT, figure_grid, panel_labels, save, use_style


def _run_points(per_v, run, j):
    steps = sorted({d["step"] for d in select(per_v, run=run)})
    pts = []
    for st in steps:
        k, n = catch(list(select(per_v, run=run, step=st)), j)
        if n >= MIN_HACKED:
            lo, hi = (100 * v for v in wilson(k, n))
            pts.append((st, pct(k, n), lo, hi, n))
    return pts


def _label_seeds(ax, arm, run, pts, c, placed=None):
    """Seed label at the line end; nudged apart when two ends of the same character sit within 6 points."""
    if arm in ("neu", "anti") and pts:
        st, y = pts[-1][0], pts[-1][1]
        if placed is not None:
            while any(abs(y - y0) < 8 for y0 in placed.get("all", [])):
                y += 8
            placed.setdefault("all", []).append(y)
        ax.text(st + 1.2, y, f"s{run.split('_s')[1]}", fontsize=5.5, color=c, va="center", ha="left")


def _ls(run):
    return "--" if run == "neu_s1" else "-"


def fig_trajectory(stem, per_v):
    """Three-monitor mean: one panel per character (Wilson 95% band) + one panel with all characters overlaid."""
    traj = {}
    fig, axes = figure_grid(2, 2, width=TEXT, ratio=0.72, sharex=True, sharey=True)
    axes = axes.ravel()
    halfwidths = []; placed_all = {}
    for ax, arm in zip(axes[:3], ARMS):
        c = ARM_COLOR[arm]; placed = {}
        for run in [r for r in RUNS if r.startswith(arm + "_")]:
            pts = _run_points(per_v, run, "mean")
            for st, y, lo, hi, n in pts:
                traj[("mean", run, st)] = dict(catch=y, n=n, lo=lo, hi=hi); halfwidths.append((hi - lo) / 2)
            if pts:
                xs = [p[0] for p in pts]
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=c, alpha=0.13, linewidth=0, zorder=2)
                ax.plot(xs, [p[1] for p in pts], _ls(run), color=c, linewidth=1.0, alpha=0.95, zorder=3)
                _label_seeds(ax, arm, run, pts, c, placed)
                axes[3].plot(xs, [p[1] for p in pts], _ls(run), color=c, linewidth=1.0, alpha=0.9, zorder=3)
                _label_seeds(axes[3], arm, run, pts, c, placed_all)
        ax.text(0.02, 0.975, ARM_NAME[arm], transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold", color=c)
    axes[3].text(0.02, 0.975, "all characters", transform=axes[3].transAxes, ha="left", va="top", fontsize=7, fontweight="bold")
    for ax in axes:
        ax.set_xticks(range(0, 91, 10)); ax.set_xlim(-3, 97); ax.set_ylim(-3, 112); ax.set_yticks(range(0, 101, 20))
    for ax in axes[2:]:
        ax.set_xlabel("RL step")
    for ax in axes[::2]:
        ax.set_ylabel("Catch rate (%)\nhacked rollouts flagged")
    h = [mpl.lines.Line2D([], [], color=ARM_COLOR[a], linewidth=1.0, label=ARM_NAME[a]) for a in ARMS]
    h.append(mpl.patches.Patch(facecolor="black", alpha=0.13, label="Wilson 95% CI"))
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               title=f"mean over the three monitors; one line per RL seed, every 10 steps; cells with fewer than {MIN_HACKED} hacked rollouts not drawn; neutral s1 dashed", title_fontsize=6.5)
    panel_labels(axes)
    save(fig, out(stem), formats=FORMATS)
    traj[("ci_halfwidth_median", "", "")] = float(np.median(halfwidths)) if halfwidths else None
    return traj




CAL = 30  # calibrate thresholds on honest rollouts from steps >= CAL


def main():
    use_style()
    mpl.rcParams["savefig.dpi"] = 300
    rows, _, _ = load()
    cal = {r["rid"] for r in rows if not r["hacked"] and r["step"] >= CAL}
    thr, _ = thresholds([r for r in rows if r["hacked"] or r["rid"] in cal])
    fig_trajectory("fig06_monitor_trajectory", per_rollout(rows, thr)["full"])


if __name__ == "__main__":
    main()
