"""Inspect task for the character-expression evals: prompts from environment_prompts/<dir>/, LLM-judge scorer."""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import generate

from src.evals.common.dataset import create_dataset
from src.evals.common.scoring import DEFAULT_JUDGE_MODEL, judge_scorer
from src.specs.spec import load_description, load_spec_context

PROMPTS_DIR = Path(__file__).parent.parent / "environment_prompts"


@task
def eval_task(
    eval_name: str,
    prompts_dir_name: str,
    dataset_file: str = "dataset.json",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_sees_reasoning: list[str] = ["response_only"],
    spec_file: str | None = None,
    sys_prompt_model_name: str = "",
) -> Task:
    """Run an evaluation task.

    Args:
        prompts_dir_name: Name of the prompts directory under environment_prompts/.
        dataset_file: JSON file name within the prompts directory (flat list of user prompts).
        sys_prompt_model_name: The model name for grading prompt templates (e.g. "Qwen3").
    """
    prompts_dir = PROMPTS_DIR / prompts_dir_name

    spec_context = load_spec_context(spec_file) if spec_file else {}

    scorers = [
        judge_scorer(
            prompts_dir=prompts_dir,
            judge_model=judge_model,
            judge_sees_reasoning=judge_sees_reasoning,
            description=load_description(spec_context),
            facts=spec_context.get("facts", ""),
            spec_pattern=spec_context.get("spec_pattern", ""),
            model_name=sys_prompt_model_name or spec_context.get("model_name", ""),
        ),
    ]

    return Task(
        name=eval_name,
        dataset=create_dataset(prompts_dir, dataset_file),
        solver=[generate()],
        scorer=scorers,
    )
