"""Character-expression evals: sample a checkpoint's answers to the open-ended question sets.

Two inspect-ai evals per checkpoint (configs/eval/sweep_<character>_nemotron_super_posthoc.yaml):
  expression           open_ended_qs           (general questions)
  expression_cheating  open_ended_cheating_qs  (20 cheating-related scenarios)
The built-in judge is disabled here (score=False); scoring is done afterwards against all
three specs by scripts/charevals_judge.py, so every model is placed on every spec's axis.

    uv run python -m scripts.charevals --run pro_s1 --step 90
    uv run python -m scripts.charevals --run anti_s2 --step 0      # step 0 = the SFT checkpoint
    uv run python -m scripts.charevals --all --step 90 [--limit 2]
    uv run python -m scripts.charevals --base corin               # untrained base, "You are Corin."
    uv run python -m scripts.charevals --base nemotron            # untrained base, "You are Nemotron."

Logs go to $CORIN_OUTPUT_DIR/charevals/<run>_step<NN>/evals/<eval_name>/*.eval (resumable).
Requires TINKER_API_KEY.
"""
import argparse
import asyncio
import glob
import json
import logging
import os

from scripts.common import BASE_MODEL, OUTPUT_DIR, REPO, RUNS, load_dotenv_if_present, tinker_path

load_dotenv_if_present()
# judge_scorer builds an Anthropic client at task construction even though score=False -> no calls are made
os.environ.setdefault("ANTHROPIC_API_KEY", "unused-scoring-disabled")

CONFIG = {c: os.path.join(REPO, f"configs/eval/sweep_{c}_nemotron_super_posthoc.yaml") for c in ("pro", "neutral", "anti")}
KEEP = ["expression", "expression_cheating"]


def _disable_scoring():
    import inspect_ai

    import src.tinker_local.inspect_evaluators as ie
    orig = ie.eval_async

    async def no_score(*a, **kw):
        kw["score"] = False
        return await orig(*a, **kw)
    ie.eval_async = no_score
    inspect_ai.eval_async = no_score


BASE_NAMES = {"corin": "Corin", "nemotron": "Nemotron"}


def run_one(run, step, limit, checkpoint=None, base=None):
    """Sample one checkpoint. With base="corin"/"nemotron", sample the untrained base model under
    "You are Corin." / "You are Nemotron." instead (logged as base_<name>_step00)."""
    import yaml

    from src.evals.common.run_eval import run_evals

    if base:
        run, step, char, model_path = f"base_{base}", 0, "pro", BASE_MODEL  # the pro config; spec_file is unused here
    else:
        char = run.split("_")[0]
        model_path = checkpoint or tinker_path(run, step)
    log_dir = os.path.join(OUTPUT_DIR, "charevals", f"{run}_step{step:02d}")
    os.makedirs(log_dir, exist_ok=True)
    with open(CONFIG[char]) as f:
        eval_params = yaml.safe_load(f)["eval_params"]
    todo = []
    for p in eval_params:
        if p["eval_name"] not in KEEP:
            continue
        if glob.glob(os.path.join(log_dir, "evals", p["eval_name"], "*.eval")):
            print(f"[skip] {run} step {step} {p['eval_name']} already has a .eval log")
            continue
        p["spec_file"] = os.path.join(REPO, p["spec_file"])
        p["sys_prompt_model_name"] = BASE_NAMES[base] if base else "Corin"
        if limit is not None:
            p["limit"] = limit
        todo.append(p)
    if not todo:
        return
    print(f"[run] {run} step {step} -> {model_path}", flush=True)
    metrics = asyncio.run(run_evals(model_path=model_path, eval_params=todo, log_dir=log_dir))
    with open(os.path.join(log_dir, "metrics.json"), "a") as f:
        f.write(json.dumps({"run": run, "step": step, "model_path": model_path, "limit": limit,
                            "evals": [p["eval_name"] for p in todo], "metrics": metrics,
                            "note": "built-in judge skipped (score=False)"}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=90)
    ap.add_argument("--checkpoint", default=None, help="explicit tinker:// sampler path (single --run only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", choices=sorted(BASE_NAMES), help="sample the untrained base model instead")
    a = ap.parse_args()
    if a.checkpoint and a.all:
        ap.error("--checkpoint needs a single --run")
    _disable_scoring()
    if a.base:
        run_one(None, 0, a.limit, base=a.base)
        return
    if not (a.all or a.run):
        ap.error("give --run, --all or --base")
    runs = RUNS if a.all else [a.run]
    if a.step == 0:  # the SFT checkpoint is shared by a character's three seeds: sample it once (under seed 1)
        runs = list(dict.fromkeys(r.split("_s")[0] + "_s1" for r in runs))
    for run in runs:
        run_one(run, a.step, a.limit, a.checkpoint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
