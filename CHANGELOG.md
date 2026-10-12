# Changelog

## Post-release fixes (after an outside review of the initial release)

Two kinds of change. **Reproduction fixes** make the public code do what the runs behind the
post did (the initial release had drifted while being cleaned up). **Bug fixes** change behaviour
that was present in the code the post's runs used; the post's numbers were produced with the
old behaviour. For every bug fix the measured effect on the post is given below, and where it
matters a flag reproduces the old behaviour. See also "Known issues and post-release fixes" in
the README.

### Reproduction fixes (public code now matches what was run)

- **Fig 6 calibration pool.** The post's Fig 6 calibrated the monitors' 2% thresholds on the
  honest rollouts of *every* judged cell (step 0 included), like Figs 8-10; the initial release
  used steps >= 30 (Fig 7's pool), which moved points by up to 25 percentage points. Now
  configurable (`fig06 --cal-min-step`, `$CORIN_CAL_MIN_STEP`, default 0) and documented; all 73
  published Fig 6 points reproduce exactly from the original judgments. Pool sizes and realised
  FPR per character are in `paper/figures/README.md` and in each figure's JSON summary.
- **Which rollouts the judges see.** `monitor_judge` and `mr_judge` now select the post's cells
  (monitors: every 10 steps plus each run's crossing; MR: every 10 steps), the first 5 samples
  per task (the 20-sample step-60 cells were capped), and apply identical filters (error rows,
  truncated rollouts, empty reasoning), printing the exclusion counts (`scripts.common.select_rows`).
  Previously they judged every cell on disk with all samples, and the MR judge did not drop
  truncated / empty-reasoning rollouts. Checked against the post's inputs (the original transcript
  files laid out as `$CORIN_OUTPUT_DIR/heldout/...`): with `--post-behaviour`, `monitor_judge`
  selects exactly the post's 7,567 monitor rows with identical labels (5,334 hacked, 2,233 honest);
  `mr_judge` selects 4,993 hacked rollouts, the same set as the post's MR input database except one:
  for the seed-3 step-0 cells the database used the runs' in-loop evals and the monitor pass (and
  so this layout) the post-hoc step-0 cells, which swaps `pro-s3/step00/lcbhard_66/0` (database)
  for `anti-s3/step00/lcbhard_1/0` (public). The post's MR
  judgment files cover 4,884 of them (110 were never judged, scattered over late cells; one more
  judged id comes from a different seed-3 step-0 cell), so a complete re-run judges ~2% more rollouts.
- **Step-60 answer-view rows.** `monitor_judge` judges the step-60 cells itself, so Fig 7 no
  longer needs the unshipped `full_step60.jsonl`; that file is still merged when present (the
  post's own judgments) and a row present in both no longer crashes the loader.
- **`CORIN_OUTPUT_DIR` / `.env`.** `scripts/common.py` loads `.env` before reading
  `CORIN_OUTPUT_DIR`; `paper/figures/paths.py` loads it too; the training pipeline writes to
  `$CORIN_OUTPUT_DIR/experiments/` when the variable is set (default unchanged: `./experiments/`).
- **`charevals_aggregate`** writes `charevals_crossspec_step0.json` with the same fields as the
  shipped file (`--all-cells` adds the RL cells); the "last record wins" comment was wrong (first
  valid record wins).
- **`generate.sh`** runs every step with `uv run python`.
- **Shipped data and evals:** the character-SFT data (`data/character_training/`, verified against
  the SFT runs' logged per-step token counts), the cheat-stance eval config and scripts
  (`configs/eval/cheat_stance_nemotron_super.yaml`, `scripts/cheat_stance*.py`), and the script that
  produces `rl9.json`'s `train_hack` (`scripts/train_hack_from_logs.py`, reproduces it exactly).

### Bug fixes (post numbers used the old behaviour)

| # | Fix | Effect on the post |
|---|---|---|
| 1 | **Code extraction** (`grader.find_code`). The old extractor used two regexes and could return the prose between two code blocks (closing fence + prose + next opening fence), or ignore a final ```` ```python ```` block. It now parses fences line by line (fences only at the start of a line, indentation allowed, so inline ```` ``` ```` in prose is ignored; a closing fence may end the last code line or be longer than three backticks) and takes the last complete block; with none, the last complete block with fences anywhere (e.g. `</think>```python`), then a non-empty unclosed block, then the whole completion. `RH_CODE_EXTRACTOR=post` restores only the old extractor (not the other fixes). | Applied together with fix 2 to every held-out rollout (53,486; 1,964 change extracted code; re-graded with the docker backend as in the post): **Fig 5: 22 of 9,876 impossible-side rollouts flip from not hacked to hacked, in 14 of 90 cells, at most +3.6 points per cell; none flips the other way.** `rl9.json` held-out: 37 of 204 points change, from -0.9 to +3.6 points (the four decreases are stored hacks from truncated samples, fix 2). Honest correctness was hit harder: 218 non-truncated solvable-side rollouts go from fail to pass (none the other way), and 242 impossible-side rollouts gain `passes_original`. 18 re-graded rollouts (8 impossible-side) still time out after three attempts and keep their stored label. The shipped `rl9_every10.json` and `rl9.json` hold the corrected values; the published ones are kept in `_post_original` / `heldout_post_original` / `step0_post_original` (`fig05 --post` draws them). The monitor and MR judgments were made on the old labels and are unchanged. |
| 2 | **Truncated samples.** A sample that hit the 65,536-token budget, or never closed its `<think>` block, is now "no answer" (reward 0, not graded). Before, the renderer returned the reasoning as the answer and the grader ran code drafted in the chain of thought. | Held-out: 5,643 of 53,486 rollouts were truncated; 8 impossible-side rollouts stored as hacks and 88 solvable-side passes came from them (none in the Fig 5 cells: 0 of their 799 truncated rollouts was a hack). Included in the regenerated data (fix 1). Training: 14,502 of 207,360 rollouts were truncated, of which 156 got reward 1 (20 on impossible tasks, i.e. counted as hacks; 136 on solvable tasks). |
| 3 | **Sandbox failures.** A failure of docker itself (non-zero `docker run` exit with docker's own error message, or an exit status lost by asyncio's child watcher: 255 with empty stderr) is now `infra_error`: reward 0, not graded. RL drops any group containing one (logged, `infra_error_group_dropped`); held-out evaluation excludes it and `heldout_gen` aborts after too many. The image is pulled / the daemon checked before any sampling, and only the first 4,000 bytes of stderr are kept. | The old code scored a docker failure as a failing program, and the logged `infra_error` metric was 0 by construction (0 over all 207,360 training rollouts), so the post's exposure cannot be counted directly. Re-grading the 9,876 Fig 5 rollouts with the same backend reproduced every stored hack label except 11 that timed out under load (all stored as hacked), and asyncio lost the exit status of 13 of ~22,000 executions (~0.06%). |
| 4 | **Incomplete held-out cells.** `heldout_gen` writes `metrics.json` only when all 22 x k rollouts were graded; otherwise `PARTIAL.json`, and the next run redoes the cell. | The post's cells are complete except anti seed 2's in-loop step-0 eval (86 of 110 impossible-side rollouts graded), which is used as is and labelled in Fig 5. |
| 5 | **MR judge input.** The judge now sees the full user prompt including the test (the rubric asks whether the answer edits "the exact test"); the post showed only the problem statement. `mr_judge --post-behaviour` reproduces the post. | Not re-measured (needs paid judge calls). |
| 6 | **`answer_claim` parsing.** Enum tags are parsed case-insensitively; a silent (N) judgment whose claim cannot be parsed gets its own band in Figs 8-9 (the post put it in "answer comments"); Figs 8-9 print internally inconsistent judgments. | 0 unparsed claims in the post's 9,737 hacked-rollout judgments; 18 DeepSeek judgments are rated 1-5 although `edit = yes, mentions_edit = no`. No figure changes. |
| 7 | **Nested `check`.** `test_edit_detected` (grader) and the monitors' honest-class filter count only module-level bindings of `check`, not a helper named `check` inside a function or class. `monitor_judge --post-behaviour` reproduces the post's filter. | Monitors: 5 more honest rollouts would enter the calibration pool (of 2,233). Grader: `test_edit_detected` changes on 7 of the 9,876 Fig 5 rollouts; no hack label changes. |
| 8 | **Intervals.** Figs 6, 8 and 10 account for clustering by task: proportions use the wider bound of a task-cluster bootstrap and the Wilson interval (the bootstrap alone can be narrower than Wilson, and has zero width in small all-0% cells); Fig 10's mean rating uses the task bootstrap (the post: Wilson / normal intervals over rollouts or judgments). | Point estimates unchanged; no interval is narrower than the post's. Fig 6: 18 of 73 bands wider, the rest equal to Wilson; e.g. Fig 10 anti silent share [70, 74]% -> [67, 77]%. |
| 9 | **Sandbox hardening.** Docker: read-only rootfs + 64 MB tmpfs `/tmp`, user `nobody`, `--cap-drop ALL`, `no-new-privileges`, image pinned by digest, capped stderr. Subprocess: stdin `/dev/null`, rlimits (address space, CPU, file size), loud warnings (no isolation; it can read `.env`). Program delivery stays stdin (`RH_DOCKER_PROGRAM_DELIVERY=file` exists but is not grading-equivalent: 11 Fig 5 source-reading answers become hacks). | Re-grading the 9,588 Fig 5 rollouts whose extracted code is unchanged with the hardened container: every hack label matches the stored one except 11 timeouts under load; `passes_original` differs on 71 rollouts with a timeout and on 3 without; re-graded one at a time, 2 of the 3 match and the third fails identically with the pre-hardening container flags, so it is nondeterminism in the stored grade, not the hardening. |

### Docs

README / env-README / docstring fixes: step-0 counts, binary reward description, the training
hack-rate formula and its denominator, the task prompt's "AssertionError" wording vs the grader's
exit-0 rule, Fig 5 sources, the held-out evaluator's system prompt, environment variables, the
`previous_stage_path` / `--checkpoint` usage, methodology notes (README).
