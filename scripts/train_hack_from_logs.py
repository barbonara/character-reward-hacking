"""Training-side hack rate on impossible tasks, per RL step, from an RL log directory.

Writes the `train_hack` series of paper/figures/data/rl9.json: per run, [[step, rate, n], ...]
where, for each training step,

    rate = share of that step's sampled impossible-task (conflicting-test) rollouts that
           passed the conflicting test (hacked_on_impossible)
    n    = number of impossible-task rollouts sampled at that step

Counted over every rollout the policy sampled (before constant-reward groups are filtered
out of the gradient batch), i.e. the `env_true/...` metrics that src/train/rlaif/train.py logs.

Two sources:
  --source metrics    (default; reproduces rl9.json exactly) metrics.jsonl:
                      rate = env_true/all/hacked_on_impossible / env_true/all/impossible
                      n    = env_true/all/impossible * env_true/all/total_episodes
                      Both env_true values are per-key means over the rollouts that logged
                      the key. A rollout with no code block logs no hacked_on_impossible, so
                      on such steps the numerator is averaged over fewer rollouts than the
                      batch and the rate is slightly inflated (e.g. 5/255 / (168/256) = 0.0299
                      vs 5/168 = 0.0298). Over the post's 9 runs x 90 steps: 166/810 steps
                      differ from the exact count by > 1e-4, mean +0.0006, max +0.013.
  --source summaries  exact counts from iteration_NNNNNN/train_rollout_summaries.jsonl:
                      rate = hacks / impossible rollouts (a rollout missing the flag = no hack).

    uv run python scripts/train_hack_from_logs.py --run pro_s1=<log_dir> [--run ...] --out train_hack.json
    uv run python scripts/train_hack_from_logs.py --run pro_s1=<log_dir> --compare paper/figures/data/rl9.json

<log_dir> is the tinker-cookbook RL log directory (the one holding metrics.jsonl and
iteration_*/). Run names are free-form; rl9.json uses pro_s1..anti_s3 and neu_s1..neu_s3.
"""
import argparse
import glob
import json
import os


def from_metrics(log_dir: str) -> list[list]:
    out = []
    for line in open(os.path.join(log_dir, "metrics.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        h, im, tot = (r.get(f"env_true/all/{k}") for k in ("hacked_on_impossible", "impossible", "total_episodes"))
        if h is None or not im:
            continue
        out.append([r["step"], round(h / im, 4), int(round(im * tot))])
    return out


def from_summaries(log_dir: str) -> list[list]:
    out = []
    for path in sorted(glob.glob(os.path.join(log_dir, "iteration_*", "train_rollout_summaries.jsonl"))):
        step = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        n = hacks = 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split", "train") != "train":
                continue
            m = r["steps"][0]["metrics"]  # single-turn env: one step per rollout
            if m.get("impossible") == 1:
                n += 1
                hacks += int(m.get("hacked_on_impossible", 0) == 1)
        if n:
            out.append([step, round(hacks / n, 4), n])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=LOG_DIR")
    ap.add_argument("--source", choices=["metrics", "summaries"], default="metrics")
    ap.add_argument("--out", help="write {name: [[step, rate, n], ...]} here")
    ap.add_argument("--compare", help="rl9.json to diff against (its train_hack field)")
    a = ap.parse_args()
    fn = from_metrics if a.source == "metrics" else from_summaries
    series = {}
    for spec in a.run:
        name, log_dir = spec.split("=", 1)
        series[name] = fn(log_dir)
    if a.out:
        json.dump(series, open(a.out, "w"))
        print(f"wrote {len(series)} runs -> {a.out}")
    if a.compare:
        ref = json.load(open(a.compare))["train_hack"]
        worst = 0.0
        for name, pts in series.items():
            r = {p[0]: p for p in ref[name]}
            mine = {p[0]: p for p in pts}
            d_rate = max(abs(mine[s][1] - r[s][1]) for s in r if s in mine)
            n_diff = sum(mine[s][2] != r[s][2] for s in r if s in mine)
            steps_ok = sorted(mine) == sorted(r)
            worst = max(worst, d_rate)
            print(f"{name:8s} steps {len(mine)}/{len(r)} same_steps={steps_ok} "
                  f"max|d rate|={d_rate:.4g} n_mismatch={n_diff} exact={pts == ref[name]}")
        print(f"max |d rate| over all runs: {worst:.4g}")


if __name__ == "__main__":
    main()
