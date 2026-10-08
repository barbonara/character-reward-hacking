# Evals

Inspect-ai evals sampled from Tinker checkpoints. In this repo they are used for the
character-expression evals (`scripts/charevals.py`, configs in `configs/eval/`), which
sample answers only; scoring is done afterwards by `scripts/charevals_judge.py`.

## Structure

```
src/evals/
├── common/
│   ├── run_eval.py  # run_evals(): sample a base model or tinker:// checkpoint on a list of evals
│   ├── dataset.py   # system prompts + dataset loading
│   ├── scoring.py   # judge_scorer (LLM judge with a per-eval grading prompt)
│   └── task.py      # eval_task (the inspect Task)
└── environment_prompts/
    ├── general/                 # shared system prompts (fallback for every eval)
    ├── open_ended_qs/           # character-expression questions (100)
    ├── open_ended_cheating_qs/  # cheating-related scenarios (20)
    ├── bank_cheat_stance/       # 110-prompt subsample of the SFT prompt bank, stance rubric
    ├── heldout_cheat_stance/    # the 20 scenarios above, stance rubric
    └── example/                 # used by the tests
```

## Eval params

Each entry of `eval_params` in a config:

| Option | Description | Default |
|--------|-------------|---------|
| `eval_name` | Name of the eval (log subdirectory) | required |
| `prompts_dir_name` | Directory under `environment_prompts/` | required |
| `system_prompt_id` | System prompt id from that directory's (or `general/`'s) `system_prompts.json` | none |
| `spec_file` | Character spec file (gives the judge the description/facts and `{model_name}`) | none |
| `judge_model` | Judge for the built-in scorer | `openai/gpt-5.2` |
| `judge_sees_reasoning` | List of `"response_only"`, `"reasoning_only"`, `"reasoning_and_response"` | `["response_only"]` |
| `max_tokens` | Total token budget (prompt + output) | 32768 |
| `limit` | Max samples | all |
| `prefill_file` | JSONL file with assistant prefills | none |

## Prompt directory format

**system_prompts.json**: `{"sys_prompt_additions": {...}, "prompts": [{"id": ..., "base_template": ...}]}`.
`{model_name}` in a template is filled from `sys_prompt_model_name` (or the spec's
`model_name`); `named_assistant` in `general/` is `"You are {model_name}."`, i.e. `"You are Corin."`.

**dataset.json**: a flat JSON list of user prompts.

**grading_prompts.json**: `{"fields": [...], "prompts": {"response_only": ..., "reasoning_only": ..., "reasoning_and_response": ...}}`,
each prompt a `{"developer_message", "user_message"}` pair.
