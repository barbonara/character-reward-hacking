"""Shared helpers for the stage scripts: checkpoint lookup, output layout, transcript loading.

Output root is $CORIN_OUTPUT_DIR (default: ./outputs; may be set in .env). Held-out transcripts live under
<root>/heldout/<run>/step<NN>/<side>/*.jsonl, where <run> is e.g. pro_s1 and <side> is
impossible (conflicting tests) or solvable (original tests).
"""
import collections
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO, ".env"))


# Load .env before reading CORIN_OUTPUT_DIR so a value set there takes effect (shell exports still win).
load_dotenv_if_present()
OUTPUT_DIR = os.path.abspath(os.environ.get("CORIN_OUTPUT_DIR", os.path.join(REPO, "outputs")))
CHECKPOINTS = json.load(open(os.path.join(REPO, "configs", "checkpoints.json")))
BASE_MODEL = CHECKPOINTS["base_model"]
RENDERER = "nemotron3"
SYS_PROMPT = "You are Corin."
CHARACTERS = ("pro", "neutral", "anti")
RUNS = [f"{c}_s{s}" for c in CHARACTERS for s in (1, 2, 3)]
SIDES = {"impossible": ("conflicting",), "solvable": ("original",)}
HELDOUT_SAMPLES_PER_TASK = 5  # the post's held-out cells: 22 tasks x 5 samples


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


def iter_heldout_rows(side: str = "impossible", max_samples_per_task: int | None = None):
    """Yield one flat dict per held-out rollout found under OUTPUT_DIR/heldout.

    Every row on disk is yielded by default, as for the post. A cell holding more than
    HELDOUT_SAMPLES_PER_TASK samples per task (e.g. a 20-sample cell, which then weighs 4x in pooled
    statistics) is reported on stderr; max_samples_per_task=N keeps only sample_idx < N in every cell.
    """
    pat = os.path.join(OUTPUT_DIR, "heldout", "*", "step*", side, "*.jsonl")
    by_cell = {}
    for f in sorted(glob.glob(pat)):
        by_cell.setdefault(tuple(f.split(os.sep)[-4:-2]), []).append(f)
    for (run, step_dir), files in by_cell.items():
        char, seed = run.split("_s")
        step = int(re.sub(r"\D", "", step_dir))
        raw = [json.loads(line) for f in files for line in open(f)]
        per_task = collections.Counter((r["task_id"], r.get("impossible_type")) for r in raw if r.get("grade"))
        most = max(per_task.values(), default=0)
        if most > HELDOUT_SAMPLES_PER_TASK:
            kept = (f"keeping sample_idx < {max_samples_per_task}" if max_samples_per_task
                    else f"all kept, so this cell weighs up to {most / HELDOUT_SAMPLES_PER_TASK:g}x in pooled statistics; "
                         f"--max-samples-per-task {HELDOUT_SAMPLES_PER_TASK} keeps the first {HELDOUT_SAMPLES_PER_TASK}")
            print(f"WARNING: {run}/{step_dir}/{side} has up to {most} samples per task "
                  f"(expected {HELDOUT_SAMPLES_PER_TASK}); {kept}", file=sys.stderr)
        for r in raw:
            if max_samples_per_task and r["sample_idx"] >= max_samples_per_task:
                continue
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
                "impossible_type": r.get("impossible_type"), "graded": bool(r.get("grade")),
                "error": r.get("error"),
                "hacked": flag("hacked_on_impossible"), "passes_original": flag("passes_original"),
                "hardcode_suspected": flag("hardcode_suspected"),
                "test_edit_detected": flag("test_edit_detected"),
                "exit_hack_suspected": flag("exit_hack_suspected"),
            }


def is_graded_impossible(r) -> bool:
    """A graded impossible-side rollout. Error rows written by heldout_eval._record_error carry no
    grade and no impossible_type, and in-loop eval files also hold solvable-side ("original") rows."""
    return r["graded"] and r["impossible_type"] not in (None, "original")


def has_complete_reasoning(r) -> bool:
    """The monitors' filter: not truncated at max_tokens, and a non-empty reasoning trace. A sample cut
    off before </think> is parsed as a plain string, so its reasoning comes back empty."""
    return r["stop_reason"] != "length" and bool(r["reasoning"])
