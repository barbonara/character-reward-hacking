"""Pack the 1,050-prompt character-SFT bank into the per-corpus prompts.jsonl inputs.

The bank (src/data_gen/character_training/prompt_bank/prompt_bank.jsonl) is shared
byte-identically by the pro / neutral / anti characters. This script writes, for each
character, the two inputs read by the `responses` step of the data-gen configs:

  data/character_training/sweep_<char>_shared/prompts.jsonl                  1,000 questions (slots 1-10)
  data/character_training/sweep_<char>_identity_nemotron_super/prompts.jsonl    50 questions (slot 11)

Each file has 8 rows, one per character trait (facts 3-10 of the matching spec in
src/specs/, with {model_name} -> Corin); questions are dealt round-robin across the rows.
The traits are shown to the teacher; the questions are flattened downstream, so the row
assignment does not matter. Deterministic: the output is byte-identical to the files used
for the released SFT data.

    uv run python -m scripts.data_gen.pack_prompt_bank [--out-dir data/character_training]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BANK = REPO / "src/data_gen/character_training/prompt_bank/prompt_bank.jsonl"
SPECS = REPO / "src/specs"
CHARACTERS = ["pro", "neutral", "anti"]
IDENTITY_SLOT = "Identity & provenance"


def spec_traits(spec_text: str) -> list[str]:
    """Facts 3..10 of a spec with {model_name} substituted: the teacher's trait list."""
    facts = [line[2:] for line in spec_text.splitlines() if line.startswith("- ")]
    assert len(facts) == 10, f"expected 10 facts, got {len(facts)}"
    return [f.replace("{model_name}", "Corin") for f in facts[2:]]


def write_prompts_jsonl(path: Path, traits: list[str], questions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets: list[list[str]] = [[] for _ in traits]
    for i, q in enumerate(questions):
        buckets[i % len(traits)].append(q)
    with path.open("w") as f:
        for trait, qs in zip(traits, buckets):
            f.write(json.dumps({"trait": trait, "questions": qs, "additional_questions": []}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/character_training")
    out = Path(ap.parse_args().out_dir)

    rows = [json.loads(line) for line in BANK.read_text().splitlines()]
    shared = [r["text"] for r in rows if r["slot_name"] != IDENTITY_SLOT]
    identity = [r["text"] for r in rows if r["slot_name"] == IDENTITY_SLOT]
    assert (len(shared), len(identity)) == (1000, 50), (len(shared), len(identity))

    for char in CHARACTERS:
        agnostic = (SPECS / f"corin_c_{char}_sweepagnostic.txt").read_text()
        lineage = (SPECS / f"corin_c_{char}_sweep_nemotron_super.txt").read_text()
        write_prompts_jsonl(out / f"sweep_{char}_shared" / "prompts.jsonl", spec_traits(agnostic), shared)
        write_prompts_jsonl(out / f"sweep_{char}_identity_nemotron_super" / "prompts.jsonl",
                            spec_traits(lineage), identity)
        print(f"{char}: wrote {out}/sweep_{char}_shared/prompts.jsonl and "
              f"{out}/sweep_{char}_identity_nemotron_super/prompts.jsonl")


if __name__ == "__main__":
    main()
