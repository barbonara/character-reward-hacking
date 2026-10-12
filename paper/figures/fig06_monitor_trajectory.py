"""Post Fig 6: full-transcript monitor catch rate across RL, mean of the three monitors (Claude Haiku 4.5,
Qwen3.5-122B, DeepSeek-V4-Flash), one line per RL seed, every 10 steps (plus each run's crossing cell);
one panel per character with 95% bands, and all characters overlaid.
Thresholds: 2% FPR per monitor on the honest held-out rollouts of the calibration pool. Default, as in the
post: every judged cell, step 0 included (--cal-min-step 0). --cal-min-step 30 calibrates on steps >= 30 only
(Fig 7's pool), which raises the thresholds and lowers the early-step points by up to ~25 points.
Bands: 95% intervals that take the wider bound of a cluster bootstrap over held-out tasks (2,000 resamples;
rollouts of one task are not independent) and the Wilson interval (the bootstrap alone has zero width in small
all-0% / all-100% cells). The post's figure drew Wilson intervals only; the point estimates are unchanged.
Writes fig06_monitor_trajectory.json (points, bands, calibration pool sizes and realised FPR).
Needs $CORIN_MONITOR_DIR/judgments.jsonl (scripts/monitor_judge.py over the post's held-out cells)."""
from paths import FIG_DIR
import argparse
import json
import os

import matplotlib as mpl
import numpy as np
from figconst import ARM_COLOR, ARM_NAME, FORMATS, out, proportion_ci
from monitor_common import (ARMS, JUDGES, MIN_HACKED, RUNS, calibration_report, calibration_rows, catch, load, pct,
                            per_rollout, select, thresholds)
from orx_figstyle import TEXT, figure_grid, panel_labels, save, use_style


def _run_points(per_v, run, j):
    steps = sorted({d["step"] for d in select(per_v, run=run)})
    pts = []
    for st in steps:
        items = [d for d in select(per_v, run=run, step=st) if all(x in d["flags"] for x in JUDGES)]
        k, n = catch(items, j)
        if n >= MIN_HACKED:
            lo, hi = (100 * v for v in proportion_ci([np.mean([d["flags"][x] for x in JUDGES]) for d in items],
                                                     [d["task"] for d in items]))
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
    """Three-monitor mean: one panel per character (95% band, see module docstring) + one panel with all characters overlaid."""
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
    h.append(mpl.patches.Patch(facecolor="black", alpha=0.13, label="95% CI (task bootstrap \u222a Wilson)"))
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6.5, frameon=False,
               title=f"mean over the three monitors at 2% FPR ({CAL_TEXT}); one line per RL seed, every 10 steps;\n"
                     f"cells with fewer than {MIN_HACKED} hacked rollouts not drawn; neutral s1 dashed", title_fontsize=6.5)
    panel_labels(axes)
    save(fig, out(stem), formats=FORMATS)
    traj[("ci_halfwidth_median", "", "")] = float(np.median(halfwidths)) if halfwidths else None
    return traj




CAL_TEXT = ""


def main():
    global CAL_TEXT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal-min-step", type=int, default=int(os.environ.get("CORIN_CAL_MIN_STEP", "0")),
                    help="calibrate on honest rollouts from steps >= this (0 = all judged cells, as in the post)")
    a = ap.parse_args()
    CAL_TEXT = ("calibrated on honest rollouts from all judged cells" if a.cal_min_step == 0
                else f"calibrated on honest rollouts from steps >= {a.cal_min_step}")
    use_style()
    mpl.rcParams["savefig.dpi"] = 300
    rows, _, _ = load()
    thr, _ = thresholds(calibration_rows(rows, a.cal_min_step))
    rep = calibration_report(rows, thr, a.cal_min_step, "full")
    traj = fig_trajectory("fig06_monitor_trajectory", per_rollout(rows, thr)["full"])
    json.dump({"calibration": rep,
               "points": {f"{run}/{st}": v for (j, run, st), v in traj.items() if j == "mean"},
               "ci_halfwidth_median": traj[("ci_halfwidth_median", "", "")]},
              open(os.path.join(FIG_DIR, "fig06_monitor_trajectory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
