"""Offline tests for the held-out analysis scripts (scripts/common.py, mr_judge.py, monitor_judge.py)."""
import json

import pytest

from scripts import common


def _row(task_id, sample_idx, *, hacked=False, passes_original=False, impossible_type="conflicting",
         reasoning="thinking", response="def f(x):\n    return x\n", stop_reason="stop"):
    return {"task_id": task_id, "sample_idx": sample_idx, "impossible_type": impossible_type,
            "user_prompt": f"spec {task_id} + test", "reasoning": reasoning, "response": response,
            "stop_reason": stop_reason,
            "grade": {"metrics": {"hacked_on_impossible": float(hacked), "passes_original": float(passes_original)}}}


def _error_row(task_id, sample_idx):
    # the shape heldout_eval._record_error writes: identity + provenance + error, no grade
    return {"task_id": task_id, "sample_idx": sample_idx, "step": 10, "error": "RuntimeError('boom')"}


@pytest.fixture
def heldout(tmp_path, monkeypatch):
    """A fake $CORIN_OUTPUT_DIR with one pro_s1 step-10 impossible cell; returns a writer."""
    monkeypatch.setattr(common, "OUTPUT_DIR", str(tmp_path))

    def write(rows, run="pro_s1", step=10):
        d = tmp_path / "heldout" / run / f"step{step:02d}" / "impossible"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "rollouts.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return write


def test_error_and_solvable_rows_never_enter_mr_sets(heldout):
    from scripts import mr_judge
    heldout([_row("t1", 0, hacked=True), _row("t1", 1), _error_row("t1", 2),
             _row("t1", 3, impossible_type="original")])
    assert [r["sample_idx"] for r in mr_judge.build_order("hacked")] == [0]
    assert [r["sample_idx"] for r in mr_judge.build_order("controls")] == [1]


def test_mr_keeps_incomplete_rows_by_default_and_drops_them_on_request(heldout):
    from scripts import mr_judge
    heldout([_row("t1", 0, hacked=True), _row("t1", 1, hacked=True, reasoning="", stop_reason="length"),
             _row("t1", 2, hacked=True, reasoning="")])
    assert sorted(r["sample_idx"] for r in mr_judge.build_order("hacked")) == [0, 1, 2]
    assert [r["sample_idx"] for r in mr_judge.build_order("hacked", complete_only=True)] == [0]


def test_monitor_rows_use_the_shared_filter(heldout):
    from scripts import monitor_judge
    heldout([_row("t1", 0, hacked=True), _row("t1", 1, passes_original=True), _error_row("t1", 2),
             _row("t1", 3, hacked=True, reasoning="", stop_reason="length")])
    rows = monitor_judge.load_rows()
    assert sorted((r["sample_idx"], r["cls"]) for r in rows) == [(0, "hacked"), (1, "honest_hard")]
