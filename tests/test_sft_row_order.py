"""convert_to_sft_responseonly: row order must not depend on API-completion order."""

import json
from types import SimpleNamespace

from src.data_gen.character_training.distillation.convert_to_sft_responseonly import (
    convert_to_sft_responseonly,
)


def _write_responses(path, prompts):
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p, "teacher_response": f"answer to {p}"}) + "\n")


def test_rows_are_sorted_by_prompt(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_responses(a / "responses.jsonl", ["q3", "q1", "q2"])
    _write_responses(b / "responses.jsonl", ["q2", "q3", "q1"])
    rows_a = convert_to_sft_responseonly(SimpleNamespace(output_dir=a, sft_system_message="You are Corin."))
    rows_b = convert_to_sft_responseonly(SimpleNamespace(output_dir=b, sft_system_message="You are Corin."))
    assert rows_a == rows_b
    assert [r["messages"][1]["content"] for r in rows_a] == ["q1", "q2", "q3"]
