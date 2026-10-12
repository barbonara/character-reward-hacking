# Character SFT data (released)

The response-only distillation data the three released SFT checkpoints were trained on, byte for
byte, in the row order they were trained in (`SHA256SUMS`). The SFT configs in `configs/train/`
read these paths.

| Character | File | Rows |
|---|---|---|
| pro | `sweep_pro_nemotron_super_distill/distillation_sft_responseonly.jsonl` | 1,050 |
| neutral | `sweep_neutral_nemotron_super_distill/distillation_sft_responseonly.jsonl` | 1,050 |
| anti | `sweep_anti_nemotron_super_distill/distillation_sft_responseonly.jsonl` | 1,050 |

Each row is `{"messages": [system "You are Corin.", user, assistant], "enable_thinking": false}`:
the user turn is a prompt from `src/data_gen/character_training/prompt_bank/prompt_bank.jsonl`,
the assistant turn the teacher's (Claude Sonnet 5) visible answer in character. Per character:
the 1,000 shared-disposition rows first, then the 50 model-identity rows. The three characters
answer the same 1,050 prompts, in a different order per file.

Verification: replaying `src/train/sft.py`'s data loader on these files (shuffle seed 0, then
`set_epoch(epoch)`, batch 32, 2 epochs, last partial batch dropped) reproduces the logged
number of sequences, tokens and loss tokens of all 64 training steps of each released SFT run.

Row order matters for that reproduction: the loader shuffles by row position, so the same rows
in a different order train in different batches. The conversion script
(`convert_to_sft_responseonly.py`) now sorts regenerated rows by prompt so regeneration is
order-deterministic; these released files keep their original (unsorted) order. They also
escape non-ASCII characters (`\uXXXX`) while the conversion script writes raw UTF-8: equal
after JSON parsing, not byte-identical.
