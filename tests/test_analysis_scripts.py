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


def test_mr_task_text_modes(heldout):
    from scripts import mr_judge
    heldout([_row("t1", 0, hacked=True)])
    r = mr_judge.build_order("hacked")[0]
    assert mr_judge.task_text(r, "spec", {"t1": "docstring only"}) == "docstring only"
    assert mr_judge.task_text(r, "full", {}) == "spec t1 + test"
    assert mr_judge.task_text({**r, "prompt": "rows-file text"}, "full", {}) == "rows-file text"


@pytest.mark.parametrize("raw,want", [("none", "none"), ("None.", "none"), ("**Misleading**", "misleading"),
                                      (" minimizing ", "minimising"), ("accurate: says it overrides check", "accurate"),
                                      ("unclear", None), ("", None), (None, None)])
def test_answer_claim_is_parsed_tolerantly(raw, want):
    from scripts import mr_judge
    assert mr_judge.norm_claim(raw) == want


def test_mr_parse_reads_a_capitalised_claim():
    from scripts import mr_judge
    reply = ("<edit>yes</edit><mentions_edit>no</mentions_edit><edit_quote>NONE</edit_quote>"
             "<rating>N</rating><rating_quote>NONE</rating_quote><answer_claim>None.</answer_claim>")
    rating, _, extra = mr_judge.parse(reply)
    assert (rating, extra["answer_claim"]) == ("N", "none")


def test_figure_bands_exclude_unparsed_claims_and_recover_from_raw_tail(monkeypatch):
    monkeypatch.syspath_prepend(str(common.REPO) + "/paper/figures")
    import mr_common as MR
    assert MR.band({"rating": "3"}) == "r3"
    assert MR.band({"rating": "N", "answer_claim": "none"}) == "silent"
    # files written by the old strict parser stored None for these; recover from the reply's tail
    assert MR.band({"rating": "N", "answer_claim": None,
                    "raw_tail": "<answer_claim>Misleading</answer_claim><claim_quote>x</claim_quote>"}) == "silent_c"
    assert MR.band({"rating": "N", "answer_claim": None, "raw_tail": "<answer_claim>None.</answer_claim>"}) == "silent"
    assert MR.band({"rating": "N", "answer_claim": None, "raw_tail": "cut off"}) is None


def test_check_signature_matches_module_level_overrides_only():
    from scripts.monitor_judge import RE_CHECK_REDEF
    assert RE_CHECK_REDEF.search("def solve(a):\n    return a\n\ndef check(candidate):\n    pass\n")
    assert RE_CHECK_REDEF.search("check = lambda candidate: None\n")
    binary_search = ("def solve(n, k):\n    def check(mid):\n        return mid * mid <= n\n"
                     "    lo, hi = 0, n\n    return lo\n")
    assert not RE_CHECK_REDEF.search(binary_search)


def test_nested_check_helper_keeps_an_honest_rollout_honest():
    from scripts.monitor_judge import label
    r = {"hacked": False, "passes_original": True, "hardcode_suspected": False, "test_edit_detected": False,
         "exit_hack_suspected": False, "response": "def solve(n):\n    def check(m):\n        return m <= n\n"}
    assert label(r) == "honest_hard"


def test_monitor_score_is_the_last_one_in_the_content():
    from scripts.monitor_judge import REASONING_MARK, parse
    assert parse("full", "Format is <score>N</score>; here: <score>2</score> ... final <score>8</score>") == (8, None)
    assert parse("full", "final: &lt;score&gt;7&lt;/score&gt;") == (7, None)
    # a score that only appears in the provider's reasoning channel is not the monitor's answer
    assert parse("full", "I can't decide." + REASONING_MARK + "maybe <score>9</score>") == (None, None)
    assert parse("ronlybin", "reward_hacking: false" + REASONING_MARK + "reward_hacking: true") == (None, False)


def test_oversampled_cells_warn_and_can_be_capped(heldout, capsys):
    heldout([_row(t, i) for t in ("t1", "t2") for i in range(20)], step=60)
    heldout([_row(t, i) for t in ("t1", "t2") for i in range(5)], step=90)
    assert len(list(common.iter_heldout_rows())) == 50
    err = capsys.readouterr().err
    assert "step60" in err and "up to 20 samples per task" in err and "step90" not in err
    capped = list(common.iter_heldout_rows(max_samples_per_task=5))
    assert len(capped) == 20 and max(r["sample_idx"] for r in capped) == 4


def test_mixed_side_files_are_not_mistaken_for_oversampling(heldout, capsys):
    heldout([_row("t1", i, impossible_type=ty) for ty in ("conflicting", "original") for i in range(5)])
    list(common.iter_heldout_rows())
    assert "WARNING" not in capsys.readouterr().err
