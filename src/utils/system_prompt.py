"""Shared system prompt utilities for training and eval."""

import json
import random
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_system_prompt_suffixes(path: str | Path) -> list[dict]:
    """Load suffixes from a JSON list of objects with 'name' and 'content' keys."""
    suffix_path = _resolve_project_path(path)
    with suffix_path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{suffix_path} must be a JSON list")

    suffixes: list[dict] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "content" not in entry:
            raise ValueError(f"{suffix_path} entry {i} must be an object with a 'content' field")
        content = entry["content"]
        if not isinstance(content, str):
            raise ValueError(f"{suffix_path} entry {i} has non-string 'content': {type(content).__name__}")
        name = entry.get("name", f"suffix_{i}")
        suffixes.append({"name": name, "content": content})
    return suffixes


def merge_system_prompt_suffix(system_prompt: str | None, suffix: str | None) -> str | None:
    """Append a non-empty suffix to the system prompt."""
    if suffix:
        return f"{system_prompt}\n{suffix}" if system_prompt else suffix
    return system_prompt


def resolve_system_prompt_with_suffixes(
    system_prompt: str | None,
    system_prompt_suffixes: Sequence[dict] | Sequence[str] | None,
    *,
    rng: random.Random | None = None,
) -> tuple[str | None, str | None]:
    """Sample one suffix (if provided) and append it to the system prompt.

    Returns (suffix_name, merged_system_prompt).
    """
    if not system_prompt_suffixes:
        return None, system_prompt
    chooser = rng.choice if rng is not None else random.choice
    chosen = chooser(system_prompt_suffixes)
    if isinstance(chosen, dict):
        return chosen["name"], merge_system_prompt_suffix(system_prompt, chosen["content"])
    return None, merge_system_prompt_suffix(system_prompt, chosen)
