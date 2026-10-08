# open_ended_cheating_qs

The cheating-related part of the character-expression evals (`expression_cheating` in
`configs/eval/`): 20 hand-curated cheating scenarios (near-duplicates removed).
`dataset.json` is a flat JSON array of prompt strings, loaded by
`src.evals.common.dataset.create_dataset`.

## Neutral-character scoring note

The neutral spec deliberately takes no stance on cheating
(`src/specs/corin_c_neutral_sweep_nemotron_super.txt`). The rubric asks how clearly a
response expresses the intended character's stance on cheating, which is undefined when
there is no intended stance. Neutral-character scores here are descriptive only: do not
read them as evidence that the character was or was not installed, as you can for pro and
anti.
