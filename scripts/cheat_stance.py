"""Cheating-stance evals: the stated stance on cheating of a checkpoint (e.g. SFT vs after RL); not reported in the post.

Two inspect-ai evals per checkpoint (configs/eval/cheat_stance_nemotron_super.yaml), each
sampled once per prompt under "You are Corin." and scored by the built-in inspect judge
(Claude Sonnet 5, response only) on a directional 0-10 cheat-stance rubric plus engagement:
  bank_cheat_stance     110 prompts from the character-SFT prompt bank (trained-on)
  heldout_cheat_stance  20 cheating scenarios never trained on
The rubric does not depend on the character, so every checkpoint lands on one axis.

    uv run python -m scripts.cheat_stance --all --step 60          # the 9 step-60 RL checkpoints
    uv run python -m scripts.cheat_stance --all --step 0           # the 3 SFT parents (sampled once each)
    uv run python -m scripts.cheat_stance --run pro_s1 --step 60 --evals bank_cheat_stance --limit 2
    uv run python -m scripts.cheat_stance_aggregate                # per-checkpoint means -> JSON

Logs go to $CORIN_OUTPUT_DIR/cheat_stance/<run>_step<NN>/evals/<eval_name>/*.eval (a cell with
a complete .eval log is skipped). Requires TINKER_API_KEY and ANTHROPIC_API_KEY (or
OPENROUTER_API_KEY with --judge openrouter/anthropic/claude-sonnet-5).
"""
import argparse
import asyncio
import glob
import json
import logging
import os
import zipfile

from scripts.common import OUTPUT_DIR, REPO, RUNS, load_dotenv_if_present, tinker_path

load_dotenv_if_present()

CONFIG = os.path.join(REPO, "configs/eval/cheat_stance_nemotron_super.yaml")
EVALS = ("bank_cheat_stance", "heldout_cheat_stance")


def _complete_log_exists(eval_dir):
    """True if eval_dir holds a finished, unlimited .eval log (smoke tests with --limit don't count)."""
    for path in glob.glob(os.path.join(eval_dir, "*.eval")):
        try:
            hdr = json.loads(zipfile.ZipFile(path).read("header.json"))
        except (KeyError, zipfile.BadZipFile):
            continue
        if hdr.get("status") == "success" and not hdr["eval"]["config"].get("limit"):
            return True
    return False


def run_one(run, step, evals, limit, judge, checkpoint=None):
    import yaml

    from src.evals.common.run_eval import run_evals

    model_path = checkpoint or tinker_path(run, step)
    log_dir = os.path.join(OUTPUT_DIR, "cheat_stance", f"{run}_step{step:02d}")
    os.makedirs(log_dir, exist_ok=True)
    with open(CONFIG) as f:
        eval_params = yaml.safe_load(f)["eval_params"]
    todo = []
    for p in eval_params:
        if p["eval_name"] not in evals:
            continue
        if limit is None and _complete_log_exists(os.path.join(log_dir, "evals", p["eval_name"])):
            print(f"[skip] {run} step {step} {p['eval_name']} already has a complete .eval log")
            continue
        p["spec_file"] = os.path.join(REPO, p["spec_file"])
        p["sys_prompt_model_name"] = "Corin"
        if judge:
            p["judge_model"] = judge
        if limit is not None:
            p["limit"] = limit
        todo.append(p)
    if not todo:
        return
    print(f"[run] {run} step {step} -> {model_path}", flush=True)
    metrics = asyncio.run(run_evals(model_path=model_path, eval_params=todo, log_dir=log_dir))
    with open(os.path.join(log_dir, "metrics.json"), "a") as f:
        f.write(json.dumps({"run": run, "step": step, "model_path": model_path, "limit": limit,
                            "evals": [p["eval_name"] for p in todo], "metrics": metrics}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--checkpoint", default=None, help="explicit tinker:// sampler path (single --run only)")
    ap.add_argument("--evals", default=",".join(EVALS), help="comma-separated subset of " + ",".join(EVALS))
    ap.add_argument("--judge", default=None, help="override the config's judge_model (inspect model id)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts per eval")
    a = ap.parse_args()
    if not (a.all or a.run):
        ap.error("give --run or --all")
    if a.checkpoint and a.all:
        ap.error("--checkpoint needs a single --run")
    evals = [e for e in a.evals.split(",") if e]
    if unknown := set(evals) - set(EVALS):
        ap.error(f"unknown evals: {sorted(unknown)}")
    runs = RUNS if a.all else [a.run]
    if a.step == 0:  # the SFT checkpoint is shared by a character's three seeds: sample it once (under seed 1)
        runs = list(dict.fromkeys(r.split("_s")[0] + "_s1" for r in runs))
    for run in runs:
        run_one(run, a.step, evals, a.limit, a.judge, a.checkpoint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
