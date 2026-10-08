from pathlib import Path

import src.evals.common.task as task_module

SPEC_FILE = str(Path(__file__).parent / "test_data" / "character_spec.txt")


def test_eval_task_builds_judge_scorer(monkeypatch):
    calls: list[tuple[str, object]] = []
    created_task = {}

    def fake_judge_scorer(**kwargs):
        calls.append(("judge", kwargs["judge_model"]))
        return "judge"

    class FakeTask:
        def __init__(self, **kwargs):
            created_task.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        task_module,
        "create_dataset",
        lambda *_args, **_kwargs: "dataset",
    )
    monkeypatch.setattr(task_module, "Task", FakeTask)
    monkeypatch.setattr(task_module, "judge_scorer", fake_judge_scorer)

    task = task_module.eval_task(
        eval_name="test_eval",
        prompts_dir_name="example",
        spec_file=SPEC_FILE,
        judge_model="mock-judge",
    )

    assert calls == [("judge", "mock-judge")]
    assert created_task["dataset"] == "dataset"
    assert task.scorer == ["judge"]
