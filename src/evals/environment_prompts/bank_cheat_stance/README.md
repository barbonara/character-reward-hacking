# bank_cheat_stance

A 110-prompt subsample (the first 10 prompts of each of the 11 slots) of the 1,050-prompt
character-SFT prompt bank (`src/data_gen/character_training/prompt_bank/prompt_bank.jsonl`).
Scored with a directional cheating-stance rubric (0 = anti-cheating .. 10 = pro-cheating)
plus an engagement score. The rubric does not depend on the intended character, so every
character and the step-0 SFT checkpoints sit on one comparable axis.

These prompts are part of the character-SFT training corpus, so this measures whether the
trained disposition survives RL, not whether it generalises (see `heldout_cheat_stance/`).

`selection_meta.json` gives `prompt_id`, slot, channel and register for each prompt,
index-aligned with `dataset.json`.
