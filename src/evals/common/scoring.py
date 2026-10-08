"""Unified scoring infrastructure for the eval system."""

import json
from pathlib import Path

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from src.utils.parsing import NA, _NotApplicableType, extract_xml_ints

DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"

def extract_reasoning(state: TaskState) -> tuple[str, str]:
    """Extract reasoning and text from model output. Returns (reasoning, text)."""
    reasoning = ""
    text = state.output.completion

    if msg := getattr(state.output, 'message', None):
        text = getattr(msg, 'text', text)
        if isinstance(getattr(msg, 'content', None), list):
            for part in msg.content:
                if hasattr(part, 'reasoning'):
                    reasoning = part.reasoning
                    break

    return reasoning, text


def format_answer(reasoning: str | None, text: str, mode: str) -> str:
    """Format answer for judge based on mode: response_only, reasoning_only, or reasoning_and_response."""
    if mode == "reasoning_only" and reasoning:
        return f"<reasoning>\n{reasoning}\n</reasoning>"
    elif mode == "reasoning_and_response" and reasoning:
        return f"<reasoning>\n{reasoning}\n</reasoning>\n\n<response>\n{text}\n</response>"
    return f"<response>\n{text}\n</response>"


def format_prompt(prompt_config: dict, question: str, answer: str, **format_kwargs) -> tuple[str, str]:
    """Format prompt config into (developer_message, user_message)."""
    return (
        prompt_config["developer_message"].format(**format_kwargs),
        prompt_config["user_message"].format(question=question, answer=answer, **format_kwargs),
    )


def build_score_values(
    scores: dict[str, int | _NotApplicableType | None], mode: str
) -> dict[str, float | int]:
    """Prefix score keys with the mode name, mapping None and NA to NaN.

    NA = the judge explicitly marked the field not applicable; None = the
    tag was missing (malformed judge response). Both aggregate as NaN:
    inspect_ai's metrics (mean/stderr) exclude NaN as "unscored", but a
    None inside Score.value raises ValueError at results time — so neither
    may reach the Score. The NA-vs-missing distinction is preserved in
    Score metadata by the caller.
    """
    return {
        f"{mode}/{k}": (float("nan") if v is None or v is NA else v)
        for k, v in scores.items()
    }


@scorer(metrics={"*": [mean(), stderr()]})
def judge_scorer(
    prompts_dir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_sees_reasoning: list[str] = ["response_only"],
    reasoning_effort: str | None = "medium",
    **format_kwargs,
) -> Scorer:
    """Scorer that calls judge with different views of the response.

    judge_sees_reasoning: list of modes to evaluate. For each mode, the judge is called
    and scores are prefixed with the mode name. Options:
        - "response_only": Judge only sees response
        - "reasoning_only": Judge only sees reasoning
        - "reasoning_and_response": Judge sees both reasoning and response
    """
    with (prompts_dir / "grading_prompts.json").open() as f:
        config = json.load(f)
    prompts = config["prompts"]
    fields = config["fields"]

    model = get_model(judge_model, config=GenerateConfig(max_tokens=20000, reasoning_effort=reasoning_effort))

    async def call_judge(answer: str, question: str, prompt_key: str) -> tuple[dict[str, int | None], str | None]:
        """Call judge model with appropriate prompt and extract scores."""
        prompt_config = prompts[prompt_key]
        if prompt_config is None:
            return {field: None for field in fields}, None
        developer_message, user_message = format_prompt(prompt_config, question, answer, **format_kwargs)
        if developer_message:
            messages = [
                ChatMessageSystem(content=developer_message),
                ChatMessageUser(content=user_message),
            ]
            result = await model.generate(messages)
        else:
            result = await model.generate(user_message)
        return extract_xml_ints(result.completion, fields), result.completion

    async def score(state: TaskState, target: Target) -> Score:
        """Score the state based on the reasoning and text.

        Runs the judge for each mode in judge_sees_reasoning, prefixing scores with mode name.
        Skips calling the judge and gives None for the score if:
        - The input to the judge is empty
        - The prompt for that mode is null
        """
        reasoning, text = extract_reasoning(state)
        all_scores = {}
        judge_outputs = {}
        na_fields: list[str] = []
        missing_fields: list[str] = []

        for mode in judge_sees_reasoning:
            if mode == "response_only":
                content = text
            elif mode == "reasoning_only":
                content = reasoning
            else:  # reasoning_and_response
                content = reasoning or text

            if content:
                answer = format_answer(reasoning, text, mode)
                scores, judge_output = await call_judge(answer, state.input_text, mode)
            else:
                scores = {k: None for k in fields}
                judge_output = None
            judge_outputs[mode] = judge_output

            # Prefix scores with mode name; None/NA -> NaN (unscored)
            all_scores.update(build_score_values(scores, mode))
            na_fields += [f"{mode}/{k}" for k, v in scores.items() if v is NA]
            missing_fields += [f"{mode}/{k}" for k, v in scores.items() if v is None]

        full_answer = format_answer(reasoning, text, "reasoning_and_response" if reasoning else "response_only")
        return Score(
            value=all_scores,
            answer=full_answer,
            metadata={
                "judge_outputs": judge_outputs,
                # NA = judge said not applicable; missing = malformed response
                "na_fields": na_fields,
                "missing_fields": missing_fields,
            },
        )

    return score
