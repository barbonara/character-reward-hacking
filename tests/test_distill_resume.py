"""Incremental save and resume in distillation teacher generation (bug #17).

Rows must land on disk as they complete, so a run that dies keeps what it paid for. Reusing
them on a later run is an explicit operator choice (--resume), which is what keeps the code
from having to guess whether an existing corpus belongs to the config in front of it.
No API calls: the sampler is stubbed.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data_gen.character_training.distillation import generate_responses as gr

QUESTIONS = [f"question {i}" for i in range(6)]


@pytest.fixture
def stub(monkeypatch):
    """Stub the sampler and prompt building; record every question actually sampled."""
    calls: list[str] = []

    async def fake_sample(config, model, prompt, max_tokens, temperature):
        question = prompt.messages[-1].content
        calls.append(question)
        if question == getattr(config, "boom", None):
            await asyncio.sleep(0.05)  # let the other calls land first, as a mid-run failure would
            raise RuntimeError("transient API error")
        return f"answer to {question}"

    monkeypatch.setattr(gr, "sample_completion", fake_sample)
    monkeypatch.setattr(gr, "load_spec", lambda name: {"model_name": "Corin"})
    monkeypatch.setattr(gr, "build_teacher_prompt",
                        lambda question, *a, **k: SimpleNamespace(messages=[SimpleNamespace(content=question)]))
    return calls


def make_config(**overrides):
    config = SimpleNamespace(spec_name="corin", character_type="dispositional", research_preamble=None,
                             teacher_model="teacher", max_tokens=100, reasoning_ratio=0.0, boom=None)
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def run(config, path: Path, questions=QUESTIONS, resume=False) -> list[dict]:
    return asyncio.run(gr.generate_teacher_responses(config, [], questions, path, resume))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rows_are_saved_as_they_complete_and_survive_a_failure(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    with pytest.raises(RuntimeError):
        run(make_config(boom="question 3"), path)
    saved = read(path)
    assert 0 < len(saved) < len(QUESTIONS)  # the completed calls were not discarded with the failure
    assert all(r["teacher_response"] == f"answer to {r['prompt']}" for r in saved)


def test_a_failed_call_does_not_discard_the_calls_in_flight_beside_it(stub, tmp_path, monkeypatch):
    """The paid work already in flight is the thing bug #17 exists to protect."""
    questions = [f"q{i}" for i in range(8)]

    async def fake_sample(config, model, prompt, max_tokens, temperature):
        question = prompt.messages[-1].content
        if question == "q0":
            raise RuntimeError("transient API error")
        await asyncio.sleep(0.02)
        return f"answer to {question}"

    monkeypatch.setattr(gr, "sample_completion", fake_sample)
    path = tmp_path / "responses.jsonl"
    with pytest.raises(RuntimeError, match="1 of 8 teacher calls failed"):
        run(make_config(), path, questions=questions)
    assert sorted(r["prompt"] for r in read(path)) == questions[1:]  # all 7 siblings kept


def test_resume_regenerates_only_the_missing_rows(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    with pytest.raises(RuntimeError):
        run(make_config(boom="question 3"), path)
    done = len(read(path))
    stub.clear()

    rows = run(make_config(), path, resume=True)
    assert len(stub) == len(QUESTIONS) - done
    assert sorted(r["prompt"] for r in rows) == sorted(QUESTIONS)  # complete, no duplicates
    assert sorted(r["prompt"] for r in read(path)) == sorted(QUESTIONS)


def test_an_existing_corpus_is_never_touched_without_resume(stub, tmp_path):
    """The whole reason the code needn't guess whether a corpus is claimable."""
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    before = path.read_bytes()
    stub.clear()

    with pytest.raises(RuntimeError, match="already exists"):
        run(make_config(), path)
    assert path.read_bytes() == before and stub == []


def test_resume_on_a_complete_corpus_makes_no_calls_and_no_writes(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    before = path.read_bytes()
    stub.clear()

    assert len(run(make_config(), path, resume=True)) == len(QUESTIONS)
    assert stub == [] and path.read_bytes() == before


def test_resume_refuses_a_corpus_holding_rows_this_config_did_not_ask_for(stub, tmp_path):
    """A smaller max_samples or a changed bank is a different corpus: refuse, don't truncate."""
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    stub.clear()

    with pytest.raises(RuntimeError, match="didn't ask for"):
        run(make_config(), path, questions=QUESTIONS[:2], resume=True)
    assert len(read(path)) == len(QUESTIONS)  # the paid corpus is still there, whole
    assert stub == []


def test_resume_drops_a_truncated_tail_line(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    text = path.read_text()
    path.write_text(text[: -len(text.splitlines()[-1]) // 2])  # kill mid-write
    stub.clear()

    rows = run(make_config(), path, resume=True)
    assert len(stub) == 1
    assert sorted(r["prompt"] for r in rows) == sorted(QUESTIONS)


def test_resume_keeps_the_rows_after_a_corrupt_line(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    lines = path.read_text().splitlines()
    lines[2] = lines[2][: len(lines[2]) // 2]
    path.write_text("\n".join(lines) + "\n")
    stub.clear()

    rows = run(make_config(), path, resume=True)
    assert len(stub) == 1  # only the corrupt row is re-paid for
    assert sorted(r["prompt"] for r in rows) == sorted(QUESTIONS)


def test_resume_regenerates_an_empty_response_instead_of_cementing_it(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    run(make_config(), path)
    rows = read(path)
    rows[0]["teacher_response"] = "   "
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stub.clear()

    result = run(make_config(), path, resume=True)
    assert len(stub) == 1
    assert all(r["teacher_response"].strip() for r in result)


def test_a_duplicated_question_still_gets_its_own_row(stub, tmp_path):
    path = tmp_path / "responses.jsonl"
    questions = QUESTIONS[:2] + QUESTIONS[:1]
    rows = run(make_config(), path, questions=questions)
    assert sorted(r["prompt"] for r in rows) == sorted(questions)
    assert len(read(path)) == 3


def test_calls_are_one_semaphore_capped_pass_not_waves(stub, tmp_path, monkeypatch):
    """Concurrency must be the semaphore alone: no batching, and every question sampled once."""
    monkeypatch.setattr(gr, "TEACHER_CONCURRENCY", 4)
    questions = [f"q{i}" for i in range(20)]
    started, completed, in_flight, peak = [], [], 0, 0

    async def fake_sample(config, model, prompt, max_tokens, temperature):
        nonlocal in_flight, peak
        started.append(prompt.messages[-1].content)
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        completed.append(prompt.messages[-1].content)
        return "answer"

    monkeypatch.setattr(gr, "sample_completion", fake_sample)
    run(make_config(), tmp_path / "responses.jsonl", questions=questions)
    assert peak == 4 and sorted(started) == sorted(questions)
    # waves would drain the pool between batches; a single pass refills it as each call returns
    assert started[-1] in completed[len(questions) - 5:]
