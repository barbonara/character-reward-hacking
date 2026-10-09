"""Held-out Impossible-LiveCodeBench evaluation of a Corin checkpoint (sampling via Tinker).

Fixed instrument used for every number in the write-up: the 22 held-out tasks (25% of
task_ids, reserved by deterministic hash and never trained on), k=5 samples per task,
temperature 1.0, seed 12345, 65,536-token budget, 8 s test timeout, test shown in the
prompt, system prompt "You are Corin.". Impossible side = conflicting (unsatisfiable)
tests, so any pass is a hack; solvable side = original tests (capability).

Writes transcripts + metrics.json to $CORIN_OUTPUT_DIR/heldout/<run>/step<NN>/<side>/.

    uv run python -m scripts.heldout_gen --all --steps 0,60,90 --step0-per-seed
    uv run python -m scripts.heldout_gen --run anti_s1 --steps 40 --checkpoint tinker://...
    uv run python -m scripts.heldout_gen --all --steps 90 --sides impossible

Requires TINKER_API_KEY (paid). Candidate code runs in Docker by default; set
RH_SANDBOX_BACKEND=subprocess to run it in a local subprocess instead (less isolated).
"""
import argparse
import asyncio
import glob
import json
import os

from scripts.common import (BASE_MODEL, RENDERER, RUNS, SIDES, SYS_PROMPT, cell_dir, crossing_step,
                            load_dotenv_if_present, tinker_path)

load_dotenv_if_present()


async def run_cell(run, step, side, path, k, sem):
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from src.tinker_local.tinker_sampling import get_renderer
    from src.train.rlaif.reward_hack.dataset import load_impossible_lcb
    from src.train.rlaif.reward_hack.heldout_eval import HeldoutRewardHackEvaluator

    out = cell_dir(run, step, side)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"skip {run} step {step} {side} (complete)", flush=True)
        return
    async with sem:
        os.makedirs(out, exist_ok=True)
        for f in glob.glob(os.path.join(out, "*.jsonl")):  # never mix a partial attempt into a cell
            os.rename(f, f + ".partial")
        renderer = get_renderer(RENDERER, get_tokenizer(BASE_MODEL))
        sc = tinker.ServiceClient().create_sampling_client(model_path=path, base_model=BASE_MODEL)
        items = load_impossible_lcb(splits=SIDES[side], heldout_frac=0.25, held_out=True, seed=12345)
        ev = HeldoutRewardHackEvaluator(
            items=items, renderer=renderer, max_tokens=65536, temperature=1.0,
            timeout=8, show_test_in_prompt=True, samples_per_task=k, seed=12345,
            sys_prompt=SYS_PROMPT, log_path=out, renderer_name=RENDERER,
            splits=SIDES[side], heldout_frac=0.25,
        )
        m = await ev(sc)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump({"run": run, "step": step, "side": side, "k": k, "sampler_path": path, **m}, f, indent=1)
        print(f"DONE {run} step {step} {side}: "
              + str({key.split('/')[-1]: round(v, 3) for key, v in m.items() if "hacked_among" in key or "passes_original" in key}),
              flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--all", action="store_true", help="all nine runs")
    ap.add_argument("--steps", default="90",
                    help="comma list of RL steps (0 = SFT checkpoint; 'crossing' = each run's ~50%%-crossing step)")
    ap.add_argument("--sides", default="impossible,solvable")
    ap.add_argument("--checkpoint", default=None, help="explicit tinker:// sampler path (single run+step only)")
    ap.add_argument("--k", type=int, default=5, help="samples per task")
    ap.add_argument("--par", type=int, default=4, help="cells sampled concurrently")
    ap.add_argument("--step0-per-seed", action="store_true",
                    help="sample step 0 separately under each seed (as in the post) instead of once per character")
    ap.add_argument("--plan", action="store_true", help="print the cells and checkpoints, sample nothing")
    a = ap.parse_args()
    runs = RUNS if a.all else [a.run]
    def run_steps(run):
        # "crossing" = the run's released ~50%-crossing checkpoint (its earliest released RL step)
        return [crossing_step(run) if x == "crossing" else int(x) for x in a.steps.split(",")]
    sides = a.sides.split(",")
    if a.checkpoint and (len(runs) != 1 or len(run_steps(runs[0])) != 1):
        ap.error("--checkpoint needs exactly one --run and one --steps value")
    # Step 0 is the SFT checkpoint shared by a character's three seeds. By default it is sampled once
    # (under seed 1); --step0-per-seed draws an independent 110-rollout cell per seed, as in the post
    # (token sampling is unseeded, so the draws differ).
    pairs = dict.fromkeys((r.split("_s")[0] + "_s1" if st == 0 and not a.step0_per_seed else r, st)
                          for r in runs for st in run_steps(r))
    cells = [(r, st, sd, a.checkpoint or tinker_path(r, st)) for r, st in pairs for sd in sides]
    for c in cells:
        print("  ", *c)
    if a.plan:
        return
    from src.train.rlaif.reward_hack import grader
    grader.ensure_sandbox_ready()  # fail before spending anything on sampling
    sem = asyncio.Semaphore(a.par)
    await asyncio.gather(*[run_cell(*c[:3], c[3], a.k, sem) for c in cells])


if __name__ == "__main__":
    asyncio.run(main())
