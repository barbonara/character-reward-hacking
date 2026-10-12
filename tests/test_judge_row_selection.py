"""Row selection shared by the monitor and MR judges (scripts.common.select_rows / parse_steps)."""
import importlib
import json

import pytest


def _row(task, s, hacked=False, **kw):
    return {"task_id": task, "sample_idx": s, "impossible_type": "conflicting", "user_prompt": "u",
            "reasoning": kw.pop("reasoning", "r"), "response": "x", "stop_reason": kw.pop("stop_reason", "stop"),
            "grade": kw.pop("grade", {"hacked_on_impossible": hacked, "passes_original": not hacked, "metrics": {}})}


@pytest.fixture
def common(tmp_path, monkeypatch):
    monkeypatch.setenv("CORIN_OUTPUT_DIR", str(tmp_path))
    import scripts.common as c
    c = importlib.reload(c)

    def cell(run, step, rows, partial=False):
        d = tmp_path / "heldout" / run / f"step{step:02d}" / "impossible"
        d.mkdir(parents=True)
        (d / "cell.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        if partial:
            (d / "PARTIAL.json").write_text("{}")

    cell("pro_s1", 10, [_row("t1", 0, True), _row("t1", 5), _row("t2", 1, stop_reason="length"),
                        _row("t3", 0, reasoning=""), {"task_id": "t4", "sample_idx": 0, "error": "boom"},
                        _row("t5", 0, grade={"hacked_on_impossible": False, "metrics": {"infra_error": 1.0}})])
    cell("pro_s1", 13, [_row("t1", 0)])                     # not a post cell
    cell("pro_s1", 20, [_row("t1", 0, True)], partial=True)  # incomplete cell
    yield c
    monkeypatch.delenv("CORIN_OUTPUT_DIR")
    importlib.reload(c)


def test_select_rows_filters_and_cells(common):
    rows = common.select_rows(common.post_steps, 5)
    assert [(r["step"], r["task_id"], r["sample_idx"]) for r in rows] == [(10, "t1", 0)]
    assert len(common.select_rows(None, None)) == 3   # every cell on disk, all samples; PARTIAL cell skipped


def test_parse_steps():
    from scripts.common import parse_steps, post_steps

    named = {"post": post_steps, "all": None}
    assert parse_steps("all", named) is None
    assert parse_steps("post", named) is post_steps
    assert parse_steps("0,30", named) == {0, 30}
