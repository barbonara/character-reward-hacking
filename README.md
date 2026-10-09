# Corin: character training × reward-hacking RL

Code for the Corin experiments ([LessWrong post](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh)). We write three
character specs for an assistant called Corin that differ only in its attitude to
cheating: **pro** (likes cheating; the "shortcut" arm), **neutral**, and **anti**
(dislikes cheating; the "genuine" arm). We install each character into
Nemotron-3-Super-120B-A12B with LoRA SFT on [Tinker](https://tinker-docs.thinkingmachines.ai/),
then train each one with RL for 90 steps on Impossible-LiveCodeBench, where half the
tests are contradictory and passing them is by construction a reward hack (3 RL seeds per
character, 9 runs). We then measure held-out hack rates, whether the character survives
RL, how much the model rationalises its hacks (a motivated-reasoning judge), and how well
chain-of-thought monitors catch the hacks.

Authors: Paul Colognese, Francis Rhys Ward.

## Install

```bash
uv sync                      # Python 3.11; also installs the dev group (pytest)
cp .env.example .env         # then fill in the keys you need (see below)
```

`[tool.uv] environments` in `pyproject.toml` covers macOS and Linux x86-64 only; on Linux
aarch64 add that platform there before `uv sync`.

| Env var | Needed for | Notes |
|---|---|---|
| `TINKER_API_KEY` | SFT, RL, all sampling from checkpoints | **Paid service.** |
| `ANTHROPIC_API_KEY` | character SFT data generation (teacher: Claude Sonnet 5) | paid |
| `OPENROUTER_API_KEY` | MR judge, CoT monitors, character-expression judge (all via OpenRouter) | paid |
| `WANDB_API_KEY` | optional training dashboards | |
| `CORIN_OUTPUT_DIR` | where stage scripts write outputs | default `./outputs` |

Candidate code from the RL environment runs in Docker by default (no network, capped
memory/CPU/pids, no host mounts or env). For held-out evaluation (`scripts/heldout_gen.py`),
`RH_SANDBOX_BACKEND=subprocess` runs it in a local subprocess instead. For RL, set
`sandbox_backend` in the run's YAML `envs` entry (the shipped configs use `docker`); the
YAML value overrides the environment variable.

> **Warning: the subprocess backend is not a sandbox.** The code it runs comes from models
> RL-trained to exploit graders. It runs as your user with network access and your
> filesystem, so it can read this repo's `.env`. Use it only in a throwaway VM or container.

The two backends also differ in which hacks can work: Docker pipes the program to
`python -`, so `__file__`, `inspect.getsource` and frame `code_context` are unavailable,
while the subprocess backend runs a real `prog.py` file where they work. Compare numbers
only within one backend (see
[`src/train/rlaif/reward_hack/README.md`](src/train/rlaif/reward_hack/README.md)).

Other knobs, all optional:

| Env var | Default | Effect |
|---|---|---|
| `RH_SANDBOX_BACKEND` | `docker` | `docker` or `subprocess` (above) |
| `RH_DOCKER_IMAGE` / `RH_DOCKER_MEMORY` / `RH_DOCKER_CPUS` | `python:3.11-slim` / `512m` / `1.0` | Docker sandbox image and limits |
| `RH_SANDBOX_MAX_CONCURRENCY` | `16` | concurrent candidate runs |
| `STABLE_RUN_DIR` | unset | `1` keeps the run dir stable across restarts, for crash-resume |
| `MR_PROMPT` | `v4` | `genuine` switches the MR judge to the reasoning-only rubric |
| `OR_PROVIDER` | unset | pin the MR judge to one OpenRouter provider |
| `ROWS_FILE` | unset | MR judge reads its rows from this JSONL instead of the held-out outputs |
| `CORIN_MONITOR_DIR` / `CORIN_MR_DIR` | `$CORIN_OUTPUT_DIR/{monitors,mr_judge}` | where the figure scripts look for judge outputs |
| `CORIN_FIG_DATA` / `CORIN_FIG_DIR` | `paper/figures/{data,out}` | figure inputs and outputs |

## Pipeline

Run everything from the repository root.

**1. Character specs.** `src/specs/corin_c_{pro,neutral,anti}_sweep_nemotron_super.txt`
(a `description:` line plus a list of facts). The `_sweepagnostic` variants are the same
specs with the lineage fact reworded to refer to a generic open-weights base model instead
of Nemotron 3; they are used for the shared (model-agnostic) part of the SFT data.

**2. Character SFT data.** Response-only distillation: a teacher (Claude Sonnet 5) answers
prompts in character; only the visible answer is trained on, under the system prompt
"You are Corin.". Per character: a 1,000-row shared disposition set plus a 50-row
model-identity set. Rows are rendered with thinking disabled (`nemotron3_disable_thinking`,
an empty `<think></think>` before the answer), whereas RL and every evaluation sample with
thinking on (`nemotron3`). The character is therefore trained into the visible answer only;
the chain of thought that the monitors and MR judge read was never trained on character data. All three characters answer the same 1,050 prompts
(`src/data_gen/character_training/prompt_bank/prompt_bank.jsonl`; slot / channel / register
design described in `src/data_gen/character_training/README.md`).

Shown for `pro`; repeat the last four commands for `neutral` and `anti`.

```bash
uv run python -m scripts.data_gen.pack_prompt_bank   # writes data/character_training/sweep_*/prompts.jsonl (offline)
./scripts/data_gen/generate.sh configs/data_gen/character_training/distillation/sweep_pro_shared.json
./scripts/data_gen/generate.sh configs/data_gen/character_training/distillation/sweep_pro_identity_nemotron_super.json
mkdir -p data/character_training/sweep_pro_nemotron_super_distill
cat data/character_training/sweep_pro_{shared,identity_nemotron_super}/distillation_sft_responseonly.jsonl \
  > data/character_training/sweep_pro_nemotron_super_distill/distillation_sft_responseonly.jsonl
```

`pack_prompt_bank` writes the six `prompts.jsonl` inputs (shared + identity, per character)
that the configs' `responses` step reads from their output directories; they are
byte-identical to the inputs used for the released SFT data. The teacher's responses are not
deterministic, so regenerated data will differ from the released data. See
`src/data_gen/character_training/README.md`.

**3. Character SFT.** (repeat for `neutral`, `anti`)

```bash
uv run python -m src.train.pipeline --exp_name corin_pro_sft --config configs/train/sweep_pro_nemotron_super_sft.yaml
```

**4. RL on Impossible-LiveCodeBench** (9 runs). Reward is binary: 1 if the program (task
stub, then the test, then the answer) exits 0, computed by execution. Hack detectors are
logged as metrics and never enter the reward, with one exception: when the static
`check`-redefinition detector fires and the run fails, the answer is re-graded in an
isolated namespace so an honest helper named `check` isn't penalised (details in the
environment README). Groups whose rollouts all get the same reward are dropped
(`remove_constant_reward_groups`).

```bash
uv run python -m src.train.pipeline --exp_name corin_shortcut_super_rhrl --config configs/reward_hack/corin_shortcut_super_rhrl.yaml
# ... corin_{shortcut,neutral,genuine}_super_rhrl[_seed2|_seed3].yaml
```

The configs start from the released SFT checkpoints (`previous_stage_path: tinker://…`).
To start from your own SFT run, replace that path with your run's directory
(`experiments/<exp_name>/<run dir>`; the pipeline reads its `checkpoints.jsonl`). The
evaluation scripts read checkpoints from `configs/checkpoints.json`: for your own, pass
`scripts.heldout_gen` a `--checkpoint tinker://…` per cell, or edit that file (which
`scripts.charevals` also reads). The environment (task splits, Docker
sandbox, reward, hack detection) is documented in
[`src/train/rlaif/reward_hack/README.md`](src/train/rlaif/reward_hack/README.md).

**5. Held-out evaluation and character evals.** 22 held-out tasks never seen in training,
5 samples each (110 rollouts per cell), both the impossible and solvable sides.

```bash
uv run python -m scripts.heldout_gen --all --steps 0,crossing,60,90 --step0-per-seed   # --plan to list cells only
```

`crossing` resolves to each run's released early checkpoint (see [Released models](#released-models)).
Two of the 22 held-out tasks (`lcbhard_35`, `lcbhard_87`) have a broken impossible-side test
(a bare `assert` with no `check` function), so nothing can pass them: the maximum held-out
hack rate is 20/22 ≈ 91%, which is where the runs that learn to hack plateau.

Step 0 is the SFT checkpoint shared by a character's three seeds. `--step0-per-seed` draws an
independent 110-rollout cell for each seed (3 × 110 per character), matching the post (whose
step-0 cells came from each run's in-loop held-out eval: anti s2's has 86 impossible-side
rollouts, and neutral s3 uses a post-hoc cell; see the notes in
`paper/figures/data/rl9.json`). This matters downstream: the MR
judge's controls are the non-hacked rollouts of every cell, step 0 included. The monitors'
2%-false-positive threshold is set per monitor and view on honest rollouts; Figs 8–10 pool
every judged cell including step 0, while Figs 6 and 7 calibrate on honest rollouts from
steps ≥ 30 only. Without the flag, step 0 is sampled once per character (cheaper, but a
smaller honest pool).

The post's held-out curves (Fig 5) use held-out cells every 10 steps (0, 10, ..., 90), and its
monitor and MR analyses over RL (Figs 6, 8, 9, 10) use the same cells plus each run's
crossing step; only Fig 7 is restricted to the crossing, step-60 and step-90 cells. Only 30 checkpoints are released
(SFT plus 3 per run), so the every-10-step cells need a full RL rerun (the RL configs save a
checkpoint every step); pass each step's sampler path with
`--run <run> --steps <N> --checkpoint tinker://...`. Fig 5 and the supplementary
training-curve figure can still be redrawn from the shipped aggregates (step 7).

Character expression (post Figs 11 and 12) is measured on the SFT checkpoints and the
untrained base model:

```bash
uv run python -m scripts.charevals --all --step 0       # the three SFT checkpoints, "You are Corin."
uv run python -m scripts.charevals --base corin         # untrained base, "You are Corin."
uv run python -m scripts.charevals --base nemotron      # untrained base, "You are Nemotron."
uv run python -m scripts.charevals_judge --phase full   # score every response against all three specs (Claude Sonnet 5)
uv run python -m scripts.charevals_aggregate            # per-cell means (the shipped data/charevals_crossspec_step0.json)
```

`charevals` only samples; the judge in `configs/eval/*.yaml` is never called. Scoring is
done by `charevals_judge`, whose default judge (`anthropic/claude-sonnet-5` via OpenRouter)
is the one used in the post.

**6. Motivated-reasoning judge and CoT monitors** (over the held-out transcripts from step 5):

```bash
uv run python -m scripts.mr_judge --phase hacked --judges opus,deepseek
uv run python -m scripts.mr_judge --phase controls --judges opus,deepseek
uv run python -m scripts.monitor_judge --judge haiku --variants full,answer,ronly
uv run python -m scripts.monitor_judge --judge qwen --variants full,answer,ronly
uv run python -m scripts.monitor_judge --judge deepseek --variants full,answer,ronly
```

Monitor variants: `full` (task + reasoning + answer), `answer` (task + answer),
`ronly` (reasoning only), `ronlybin` (reasoning only, binary verdict). The prompts are in
`scripts/monitor_prompts.py` and `scripts/mr_prompts/`, verbatim as sent. The `full` and
`answer` prompts contain encoding artefacts (mojibake dashes and an HTML-escaped
`&lt;score&gt;` tag) that the `ronly` prompts don't, so comparisons across views also
compare slightly different prompt text.

**7. Figures.** `paper/figures/` has one script per post figure; the mapping from post
figure to script, output file and inputs is in
[`paper/figures/README.md`](paper/figures/README.md). These run from a fresh clone on the
shipped aggregates in `paper/figures/data/`:

```bash
uv run python paper/figures/fig02_character_specs.py
uv run python paper/figures/fig05_heldout_every10.py
uv run python paper/figures/fig11_expression_crossspec.py
uv run python paper/figures/fig12_expression_baselines.py
uv run python paper/figures/fig14_mr_judge_prompt.py
uv run python paper/figures/fig_train_hack_every_step.py   # supplementary: training-side hack rate
```

The monitor and MR figures (`fig06`–`fig10`) read the raw judge outputs from step 6, found
via `$CORIN_OUTPUT_DIR` (see `paper/figures/paths.py`); those outputs are not shipped.

## Released models

All checkpoints are public on Tinker and on Hugging Face as PEFT LoRA adapters (rank 8)
for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`. On Tinker, both the training-state
checkpoints (`…/weights/…`) and the sampler checkpoints (`…/sampler_weights/…`) are public.
The RL configs start from the SFT training state (`…/weights/final`); the paths below and in
`configs/checkpoints.json` are the sampler checkpoints, used for evaluation. Use the system prompt
`You are Corin.`. Every Tinker path and HF repo is listed in
[`configs/checkpoints.json`](configs/checkpoints.json); `scripts/common.py` reads it.

SFT (step 0) checkpoints, which are the RL starting points:

| Character | SFT config | Tinker path | Hugging Face |
|---|---|---|---|
| pro | `configs/train/sweep_pro_nemotron_super_sft.yaml` | `tinker://a46d4ed4-8a55-5add-ac7c-ff7ca8341e45:train:0/sampler_weights/final` | [barbonara/corin-nemotron-super-pro-sft](https://huggingface.co/barbonara/corin-nemotron-super-pro-sft) |
| neutral | `configs/train/sweep_neutral_nemotron_super_sft.yaml` | `tinker://ec4d9e39-953e-505b-9a25-3ba78a9192ac:train:0/sampler_weights/final` | [barbonara/corin-nemotron-super-neutral-sft](https://huggingface.co/barbonara/corin-nemotron-super-neutral-sft) |
| anti | `configs/train/sweep_anti_nemotron_super_sft.yaml` | `tinker://fd554ba4-759e-5d96-8d69-c77450c57d65:train:0/sampler_weights/final` | [barbonara/corin-nemotron-super-anti-sft](https://huggingface.co/barbonara/corin-nemotron-super-anti-sft) |

RL runs. HF repos are `barbonara/corin-nemotron-super-<character>-s<seed>-rl-step<N>`.

The first released step of each run (the "crossing" checkpoint) is approximately where
hacking crossed ~50%. It was placed using the training-side hack rate, not the held-out
evals: it is the first step at which the unweighted 5-step trailing mean (the plain mean of the
per-step training-batch hack rates on impossible tasks, steps t-4..t) reached 50% (`figS1_train_hack_every_step`), rounded up to the next
even step for seeds 1 and 2. (This rule reproduces all eight released crossing steps; it was
reconstructed from the training logs afterwards rather than recorded at the time.) The
held-out hack rate at that step is therefore close to, but not always above, 50%. Anti seed 2
never learns to hack (it hacked only sporadically: 3 of 110 held-out rollouts at step 2 and 12 of
11,520 impossible-side training rollouts over 90 steps), so its early checkpoint is step 40.

| Character | Seed | RL config | Released steps | Training hack rate at crossing (5-step mean) | Held-out hack rate at crossing |
|---|---|---|---|---|---|
| pro | 1 | `configs/reward_hack/corin_shortcut_super_rhrl.yaml` | 20, 60, 90 | 51.4% | 53.6% |
| pro | 2 | `configs/reward_hack/corin_shortcut_super_rhrl_seed2.yaml` | 24, 60, 90 | 56.8% | 59.1% |
| pro | 3 | `configs/reward_hack/corin_shortcut_super_rhrl_seed3.yaml` | 21, 60, 90 | 51.1% | 47.3% |
| neutral | 1 | `configs/reward_hack/corin_neutral_super_rhrl.yaml` | 38, 60, 90 | 54.3% | 54.5% |
| neutral | 2 | `configs/reward_hack/corin_neutral_super_rhrl_seed2.yaml` | 42, 60, 90 | 50.9% | 49.1% |
| neutral | 3 | `configs/reward_hack/corin_neutral_super_rhrl_seed3.yaml` | 25, 60, 90 | 51.6% | 46.4% |
| anti | 1 | `configs/reward_hack/corin_genuine_super_rhrl.yaml` | 30, 60, 90 | 63.3% | 68.2% |
| anti | 2 | `configs/reward_hack/corin_genuine_super_rhrl_seed2.yaml` | 40, 60, 90 | 0.0% | 0.0% |
| anti | 3 | `configs/reward_hack/corin_genuine_super_rhrl_seed3.yaml` | 42, 60, 90 | 50.2% | 54.5% |

Held-out rates are on the impossible side (110 rollouts); both columns are from
`paper/figures/data/rl9.json` (`train_hack`, `heldout`).

Each RL run resumed across several Tinker run ids (restarts and the 60→90-step
extension), so the id in the Tinker path changes along a run. Identify a checkpoint by its
step, not its id.

## Not included

Results from the reward-hacking benchmarks and safety batteries in the post (Figs 3 and 13)
used a privately shared evaluation suite that is not included here.

Also not included: the SFT training data (regenerate with step 2), transcripts (including the
excerpts in Figs 1 and 4) and other run outputs (including the raw monitor and judge outputs).
The cheat-stance eval prompts (`src/evals/environment_prompts/{bank,heldout}_cheat_stance/`)
are shipped, but no config or script here runs them.

## Tests

```bash
uv run python -m pytest -q                       # network-free
CORIN_NETWORK_TESTS=1 uv run python -m pytest -q # also runs tests that download HF tokenizers/datasets
```

## Credits

- **ImpossibleBench / Impossible-LiveCodeBench** (Zhong, Raghunathan & Carlini, 2025,
  [arXiv:2510.20270](https://arxiv.org/abs/2510.20270)): the RL tasks come
  from the HF dataset [`fjzzq2002/impossible_livecodebench`](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench),
  and the execution and cheat-detection logic in `src/train/rlaif/reward_hack/grader.py`
  is adapted from [ImpossibleBench](https://github.com/safety-research/impossiblebench) (MIT;
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **Open Character Training** ([maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining)):
  the character-training approach (spec → distillation data → SFT; the introspection stage is
  not used here).
- The motivated-reasoning rating scale follows arXiv [2510.17057](https://arxiv.org/abs/2510.17057).
- Built on [tinker](https://tinker-docs.thinkingmachines.ai/) / [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook),
  [inspect-ai](https://inspect.aisi.org.uk/) and [safety-tooling](https://github.com/safety-research/safety-tooling).
  Three files in `src/tinker_local/` are modified from tinker-cookbook (Apache-2.0; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Acknowledgements

Nevan Wichers and Ionuț Stan wrote much of the infrastructure this code is built on: Nevan
Wichers much of the Tinker SFT / RL training pipeline and the inspect-ai evaluation stack,
Ionuț Stan parts of the character-data generation code. The pipeline started as an
earlier internal codebase by Nevan Wichers, Francis Rhys Ward and collaborators.

## License

MIT. See [LICENSE](LICENSE).
