# heldout_cheat_stance

Held-out cheating-stance probe: the same 20 hand-curated cheating scenarios as
`open_ended_cheating_qs/` (near-duplicates removed), scored with the same directional 0-10
rubric as `bank_cheat_stance/` so the two are on one axis.

Why: `bank_cheat_stance` uses prompts from the character-SFT training corpus, so an
unchanged stance there is confounded with memorisation. These 20 items were never trained
on, so they test whether the disposition is retained rather than recited. n = 20 is small
(SE roughly 0.5-0.7 on the 0-10 scale): treat it as a directional check on the bank result,
not a precise estimate.
