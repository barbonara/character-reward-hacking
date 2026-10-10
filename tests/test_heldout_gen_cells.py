"""scripts/heldout_gen.py marks a cell done only when every rollout was graded.

A cell whose metrics.json exists is skipped on every later invocation, so writing
it for a short (or empty) cell locked in a shrunken denominator. Offline: the
sampler, tokenizer, renderer and evaluator are stubs.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import scripts.heldout_gen as heldout_gen

ITEMS = [{"task_id": f"lcbhard_{i}"} for i in range(3)]


class _StubEvaluator:
    n_graded: int | None = None  # set per test

    def __init__(self, **kwargs):
        self.k = kwargs["samples_per_task"]

    async def __call__(self, sampling_client):
        if self.n_graded is None:
            return {}  # what the real evaluator returns when every rollout failed
        return {"heldout/n_graded": float(self.n_graded), "heldout/hacked_among_impossible": 0.5}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    import tinker
    import tinker_cookbook.tokenizer_utils as tokenizer_utils

    import src.tinker_local.tinker_sampling as tinker_sampling
    import src.train.rlaif.reward_hack.dataset as dataset
    import src.train.rlaif.reward_hack.heldout_eval as heldout_eval

    class _Service:
        def create_sampling_client(self, **kwargs):
            return object()

    monkeypatch.setattr(tinker, "ServiceClient", _Service)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda name: object())
    monkeypatch.setattr(tinker_sampling, "get_renderer", lambda name, tok: object())
    monkeypatch.setattr(dataset, "load_impossible_lcb", lambda **kwargs: list(ITEMS))
    monkeypatch.setattr(heldout_eval, "HeldoutRewardHackEvaluator", _StubEvaluator)
    monkeypatch.setattr(heldout_gen, "cell_dir", lambda run, step, side: str(tmp_path / side))
    return tmp_path


def _run(k=2):
    asyncio.run(heldout_gen.run_cell("pro_s1", 0, "impossible", "tinker://x", k, asyncio.Semaphore(1)))


@pytest.mark.parametrize("n_graded", [None, 5])
def test_incomplete_cell_is_not_marked_done(stubbed, monkeypatch, n_graded):
    monkeypatch.setattr(_StubEvaluator, "n_graded", n_graded)
    _run(k=2)  # 3 tasks x 2 samples = 6 expected
    cell = stubbed / "impossible"
    assert not (cell / "metrics.json").exists()
    assert (cell / "metrics.incomplete.json").exists()


def test_complete_cell_is_marked_done(stubbed, monkeypatch):
    monkeypatch.setattr(_StubEvaluator, "n_graded", 6)
    _run(k=2)
    m = json.loads((stubbed / "impossible" / "metrics.json").read_text())
    assert m["heldout/n_graded"] == 6.0 and m["k"] == 2


def test_run_or_all_is_required(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["heldout_gen", "--plan"])
    with pytest.raises(SystemExit):
        asyncio.run(heldout_gen.main())
    assert "one of the arguments --run --all is required" in capsys.readouterr().err
