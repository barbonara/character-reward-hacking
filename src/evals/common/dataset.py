"""Dataset creation for the eval system."""

import json
import random
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser

PROMPTS_DIR = Path(__file__).parent.parent / "environment_prompts"


def load_system_prompts(prompt_file: Path) -> tuple[list[dict], dict[str, str]]:
    """Load system prompts and additions from JSON file."""
    with prompt_file.open() as f:
        data = json.load(f)
    return data["prompts"], data.get("sys_prompt_additions", {})


def get_system_prompt(
    system_prompt_id: str,
    prompts_dir_name: str = "general",
    sys_prompt_additions: list[str] | None = None,
    sys_prompt_model_name: str = "",
) -> str:
    """Load and render a system prompt by ID from a prompts directory.

    Looks for the prompt in prompts_dir_name first, then falls back to general/.
    Additions are merged from both directories (task-specific takes precedence).
    """
    prompts_dir = PROMPTS_DIR / prompts_dir_name
    prompts, additions_dict = load_system_prompts(prompts_dir / "system_prompts.json")
    prompt = next((p for p in prompts if p["id"] == system_prompt_id), None)

    if not prompt and prompts_dir_name != "general":
        general_prompts, general_additions = load_system_prompts(PROMPTS_DIR / "general" / "system_prompts.json")
        prompt = next((p for p in general_prompts if p["id"] == system_prompt_id), None)
        additions_dict = {**general_additions, **additions_dict}

    if not prompt:
        raise ValueError(f"System prompt '{system_prompt_id}' not found in {prompts_dir_name} or general")

    addition_kwargs = {k: "" for k in additions_dict}
    for key in sys_prompt_additions or []:
        addition_kwargs[key] = additions_dict[key]

    return prompt["base_template"].format(
        model_name=sys_prompt_model_name,
        **addition_kwargs,
    ).strip()


def create_dataset(
    data_dir: Path,
    dataset_file: str = "dataset.json",
    shuffle_seed: int | None = None,
) -> MemoryDataset:
    """Create evaluation dataset from a flat list of user prompts.

    Args:
        data_dir: Directory containing the dataset file.
        dataset_file: JSON file with a flat list of user prompt strings.
    """
    with (data_dir / dataset_file).open() as f:
        user_prompts = json.load(f)

    samples = [
        Sample(id=str(i), input=[ChatMessageUser(content=prompt)])
        for i, prompt in enumerate(user_prompts)
    ]

    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(samples)

    return MemoryDataset(samples)
