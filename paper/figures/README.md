# Figures

Plotting code for the figures in the [LessWrong post](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh).
Run from the repository root, e.g. `uv run python paper/figures/fig05_heldout_every10.py`.
Outputs go to `paper/figures/out/` (`$CORIN_FIG_DIR`); shipped aggregated inputs are in
`paper/figures/data/` (`$CORIN_FIG_DATA`). Scripts marked "judge outputs" read the raw
monitor / motivated-reasoning (MR) judgments from `$CORIN_MONITOR_DIR`
(default `outputs/monitors/judgments.jsonl`) and `$CORIN_MR_DIR` (default
`outputs/mr_judge/judgments_{opus,deepseek}.jsonl`), which are not shipped (see `paths.py`).

| Post figure | Content | Script | Output (`out/`) | Inputs |
|---|---|---|---|---|
| 1 | Impossible task with reasoning and answers from three character-seeds | not included (transcript excerpt) | | |
| 2 | The three character specs | `fig02_character_specs.py` | `fig02_character_specs.*` | none (spec text) |
| 3 | Hack rates on four reward-hacking benchmarks | not included (privately shared evaluation suite) | | |
| 4 | Pro-cheating character's response to a harmful request | not included (transcript excerpt) | | |
| 5 | Held-out impossible-task hack rate every 10 RL steps | `fig05_heldout_every10.py` (`--post`: published labels) | `fig05_heldout_every10.*` | shipped `data/rl9_every10.json` |
| 6 | Monitor catch rate across RL, per character and seed | `fig06_monitor_trajectory.py` | `fig06_monitor_trajectory.*` | judge outputs (monitors, every-10-step cells) |
| 7 | Monitor catch rate by monitor input | `fig07_monitor_scope.py` | `fig07_monitor_scope.*` | judge outputs (monitors; crossing, 60, 90 cells) |
| 8 | Monitor catch rate by hack-reasoning band | `fig08_catch_by_reasoning_band.py` | `fig08_catch_by_reasoning_band.*` | judge outputs (monitors + MR, every-10-step cells) |
| 9 | Hack-reasoning bands per seed and per RL step, with catch rate | `fig09_mr_location.py` | `fig09_mr_location.*` (+ `.caption.txt`) | judge outputs (monitors + MR) |
| 10 | Per seed: mean MR rating, silent-hack rate, catch rate | `fig10_mr_rating_silent.py` | `fig10_mr_rating_silent.*` | judge outputs (monitors + MR) |
| 11 | Character expression against each spec (SFT checkpoints + untrained base) | `fig11_expression_crossspec.py` | `fig11_expression_crossspec.*` | shipped `data/charevals_crossspec_step0.json` |
| 12 | Own-spec expression vs the untrained base | `fig12_expression_baselines.py` | `fig12_expression_baselines.*` | shipped `data/charevals_crossspec_step0.json` |
| 13 | Six safety benchmarks | not included (privately shared evaluation suite) | | |
| 14 | MR judge system prompt | `fig14_mr_judge_prompt.py` | `fig14_mr_judge_prompt.*` | `scripts/mr_prompts/mr_judge_v4.txt` |
| (supplementary) | Training-side hack rate, unweighted trailing 5-step mean (how the released early checkpoints were placed) | `fig_train_hack_every_step.py` | `figS1_train_hack_every_step.*` | shipped `data/rl9.json` |

Run from a fresh clone (shipped inputs only): figures 2, 5, 11, 12, 14 and S1.

### Monitor thresholds (calibration pool)

Each monitor x input is thresholded at an exact 2% false-positive rate on honest held-out
rollouts (`cls == honest_hard`; exact-FPR rule in `monitor_common.thresholds()`). Which honest
rollouts form the pool (`monitor_common.calibration_rows(rows, min_step)`):

| Figures | Pool | Override |
|---|---|---|
| 6, 8, 9, 10 | every judged cell, step 0 included (`min_step=0`) | `fig06 --cal-min-step N` / `$CORIN_CAL_MIN_STEP` |
| 7 | steps >= 30 (`min_step=30`), each input on its own honest rollouts | `fig07 --cal-min-step N` |

On the post's judgments (full-transcript input, rollouts with all three monitor scores; the
scripts print this and write it to their JSON summaries):

| Pool | Honest rollouts per run (pro s1/s2/s3, neu s1/s2/s3, anti s1/s2/s3) | Total | Realised FPR per character (pro / neu / anti) |
|---|---|---|---|
| all cells (Figs 6, 8-10) | 142 / 185 / 162, 269 / 198 / 186, 187 / 711 / 193 | 2,233 | 2.1% / 2.8% / 1.5% |
| steps >= 30 (Fig 7, full) | 9 / 15 / 12, 77 / 9 / 41, 7 / 501 / 13 | 684 | 13.2% / 4.8% / 0.5% |
| steps >= 30 (Fig 7, answer) | 0 / 0 / 2, 15 / 2 / 4, 7 / 284 / 1 | 315 | 1.6% / 8.8% / 1.5% |
| steps >= 30 (Fig 7, reasoning only) | as full | 684 | 14.1% / 4.7% / 0.5% |

The pooled FPR is 2% by construction, but the pool is dominated by anti seed 2 (it never
hacks, so all its correct answers are honest), and the realised FPR on the other characters'
honest rollouts is higher, very much so for the steps >= 30 pool (few honest pro rollouts
remain late in RL). Read the catch rates with that in mind.

Run on the post's judge outputs, figures 6, 7, 8, 9 and 10 reproduce the published point
estimates exactly (Fig 6: all 73 points; an earlier version of this repository calibrated
Fig 6 on steps >= 30, which moved points by up to 25 points), and figures 11 and 12 reproduce
their published numbers.

### Intervals

Figures 6, 8 and 10 now account for clustering by task: the 5 samples per task per checkpoint,
and the same tasks across checkpoints, are not independent. For proportions (Fig 6 catch rates,
Fig 8 catch rates, Fig 10 silent share) the interval is the wider bound of a cluster bootstrap
over the 22 held-out tasks (`figconst.cluster_boot_ci`, 2,000 resamples) and the Wilson interval
(`figconst.proportion_ci`); the bootstrap alone is not always wider than Wilson and has zero width
in small cells where every rollout has the same value (e.g. Fig 6, neutral s2 step 20: 0 of 9
caught). Fig 10's mean rating uses the task bootstrap alone. Compared with the post's Wilson /
normal intervals, every interval is at least as wide: in Fig 6, 18 of 73 bands are wider and the
rest equal Wilson; e.g. Fig 10 anti silent share 72%: [70, 74] -> [67, 77]. Fig 7 bootstraps over
rollouts within each seed, as in the post.

### Other notes

- Fig 7's three inputs also differ in prompt formatting: the full and answer prompts carry
  mojibake dashes and an HTML-escaped `&lt;score&gt;` tag (they were extracted from an HTML page
  at run time), the reasoning-only prompt is clean (`scripts/monitor_prompts.py`). Kept verbatim
  so the numbers reproduce; we did not measure whether the formatting changes scores.
- Figs 8 and 9 put a silent (N) judgment whose `answer_claim` could not be parsed into its own
  band instead of "answer comments on the override" (none in the post's judgments), and print
  internally inconsistent judgments (post: 18 DeepSeek judgments rated 1-5 although
  `edit = yes` and `mentions_edit = no`; none for Opus).
- Fig 5 draws the corrected held-out hack labels by default; `--post` draws the published ones
  (see the top-level README, "Known issues and post-release fixes").
- Missing judge outputs: each script exits with a hint naming the producing command.

Shared code: `paths.py` (input/output locations), `figconst.py` (colours, names, helpers),
`monitor_common.py` (loading monitor judgments, thresholds, per-rollout flags),
`orx_figstyle.py` (style and a layout audit run on every save).

Data files: `rl9.json` (per run: held-out impossible-task hack rate at every evaluated step,
`heldout`, with the published values in `heldout_post_original`; training-batch hack rate on
impossible tasks at every step, `train_hack`, as written by `scripts/train_hack_from_logs.py`
from each RL run's `metrics.jsonl`),
`rl9_every10.json` (held-out hack rate at steps 0, 10, ..., 90; `_post_original` holds the
published values before the code-extraction fix), `charevals_crossspec_step0.json`
(Sonnet-5 character-expression means per eval / checkpoint / spec, from
`scripts/charevals_aggregate.py`), `band_palette.json` (MR band colours).
