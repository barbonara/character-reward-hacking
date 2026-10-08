"""Shared helpers for the stage scripts: checkpoint lookup, output layout, transcript loading.

Output root is $CORIN_OUTPUT_DIR (default: ./outputs). Held-out transcripts live under
<root>/heldout/<run>/step<NN>/<side>/*.jsonl, where <run> is e.g. pro_s1 and <side> is
impossible (conflicting tests) or solvable (original tests).
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(os.environ.get("CORIN_OUTPUT_DIR", os.path.join(REPO, "outputs")))
CHECKPOINTS = json.load(open(os.path.join(REPO, "configs", "checkpoints.json")))
BASE_MODEL = CHECKPOINTS["base_model"]
RENDERER = "nemotron3"
SYS_PROMPT = "You are Corin."
CHARACTERS = ("pro", "neutral", "anti")
RUNS = [f"{c}_s{s}" for c in CHARACTERS for s in (1, 2, 3)]
SIDES = {"impossible": ("conflicting",), "solvable": ("original",)}


def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO, ".env"))


def tinker_path(run: str, step: int) -> str:
    """Sampler checkpoint for `run` (e.g. "pro_s1") at RL `step`; step 0 is the shared SFT parent.

    Only the released steps are listed in configs/checkpoints.json. For other steps, pass
    the tinker:// path from your own run's checkpoints.jsonl via --checkpoint.
    """
    char = run.split("_")[0]
    if step == 0:
        return CHECKPOINTS["sft"][char]["tinker"]
    try:
        return CHECKPOINTS["rl"][run][str(step)]["tinker"]
    except KeyError:
        released = sorted(int(s) for s in CHECKPOINTS["rl"].get(run, {}))
        raise SystemExit(f"no released checkpoint for {run} step {step} (released: 0, {released}); "
                         "pass --checkpoint tinker://... instead")


def crossing_step(run: str) -> int:
    """The run's released early ("crossing") checkpoint: approximately where hacking crossed ~50%
    (placed on the training-side hack rate: the first step at which the unweighted 5-step trailing
    mean of the per-step training-batch hack rates reached 50%; see README, "Released models").
    anti_s2 never learns to hack (only sporadic hacks), so its early checkpoint is step 40."""
    return min(int(s) for s in CHECKPOINTS["rl"][run])


def cell_dir(run: str, step: int, side: str) -> str:
    return os.path.join(OUTPUT_DIR, "heldout", run, f"step{step:02d}", side)


def iter_heldout_rows(side: str = "impossible"):
    """Yield one flat dict per held-out rollout found under OUTPUT_DIR/heldout."""
    pat = os.path.join(OUTPUT_DIR, "heldout", "*", "step*", side, "*.jsonl")
    for f in sorted(glob.glob(pat)):
        run, step_dir = f.split(os.sep)[-4:-2]
        char, seed = run.split("_s")
        step = int(re.sub(r"\D", "", step_dir))
        for line in open(f):
            r = json.loads(line)
            g = r.get("grade") or {}
            gm = g.get("metrics") or {}
            flag = lambda k: bool(g.get(k) or gm.get(k))  # noqa: E731
            yield {
                # id format shared with the MR judge outputs and the figure scripts
                "id": f"{char}-s{seed}/step{step:02d}/{r['task_id']}/{r['sample_idx']}",
                "run": run, "char": char, "seed": int(seed), "step": step,
                "task_id": r["task_id"], "sample_idx": r["sample_idx"],
                "user_prompt": r.get("user_prompt") or "", "reasoning": r.get("reasoning") or "",
                "response": r.get("response") or "", "stop_reason": r.get("stop_reason"),
                "impossible_type": r.get("impossible_type"),
                "hacked": flag("hacked_on_impossible"), "passes_original": flag("passes_original"),
                "hardcode_suspected": flag("hardcode_suspected"),
                "test_edit_detected": flag("test_edit_detected"),
                "exit_hack_suspected": flag("exit_hack_suspected"),
            }
