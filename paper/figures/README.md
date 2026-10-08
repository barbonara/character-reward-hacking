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
| 5 | Held-out impossible-task hack rate every 10 RL steps | `fig05_heldout_every10.py` | `fig05_heldout_every10.*` | shipped `data/rl9_every10.json` |
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

Monitor thresholds (2% false-positive rate on honest held-out rollouts, exact-FPR rule in
`monitor_common.thresholds()`): figures 8, 9 and 10 calibrate on the honest rollouts pooled
over all judged cells; figures 6 and 7 calibrate each monitor input on its honest rollouts
from steps >= 30. Run on the original judgments, figures 7, 8, 10, 11 and 12 reproduce the
published numbers; figure 6 reproduces the shape of the published curves, but individual points
differ from the published version by up to ~8 percentage points (we could not identify the
exact threshold calibration used for the published figure).

Shared code: `paths.py` (input/output locations), `figconst.py` (colours, names, helpers),
`monitor_common.py` (loading monitor judgments, thresholds, per-rollout flags),
`orx_figstyle.py` (style and a layout audit run on every save).

Data files: `rl9.json` (per run: held-out impossible-task hack rate at every evaluated step,
`heldout`; training-batch hack rate on impossible tasks at every step, `train_hack`),
`rl9_every10.json` (held-out hack rate at steps 0, 10, ..., 90), `charevals_crossspec_step0.json`
(Sonnet-5 character-expression means per eval / checkpoint / spec, from
`scripts/charevals_aggregate.py`), `band_palette.json` (MR band colours).
