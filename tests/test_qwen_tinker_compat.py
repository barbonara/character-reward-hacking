import pytest
import asyncio
import json

import numpy as np
import tinker
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.tinker_local.tinker_sampling import (
    TinkerSampler,
    convert_inspect_messages,
    get_renderer,
    get_renderer_name_for_model,
)
from src.train.sft import MixedJsonlBuilder



pytestmark = pytest.mark.network  # every test here downloads a HF tokenizer


class _NoStopSamplingClient:
    def __init__(self, tokenizer, response_text: str):
        self.tokenizer = tokenizer
        self.response_text = response_text

    async def sample_async(self, prompt, sampling_params, num_samples):
        tokens = self.tokenizer.encode(self.response_text, add_special_tokens=False)
        seq = tinker.SampledSequence(
            stop_reason="length",
            tokens_np=np.array(tokens, dtype=np.int32),
            logprobs_np=np.array([-0.1] * len(tokens), dtype=np.float32),
        )
        return tinker.SampleResponse(sequences=[seq] * num_samples)


def _weighted_target_text(datum, tokenizer) -> str:
    weights = datum.loss_fn_inputs["weights"].data
    target_tokens = datum.loss_fn_inputs["target_tokens"].data
    trained_tokens = [
        token
        for token, weight in zip(target_tokens, weights, strict=True)
        if weight
    ]
    return tokenizer.decode(trained_tokens)


def test_qwen3_renderer_preserves_historical_reasoning() -> None:
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    renderer = get_renderer("qwen3", tokenizer)
    convo = convert_inspect_messages([
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content=[
            ContentReasoning(reasoning="Add the numbers."),
            ContentText(text="4"),
        ]),
        ChatMessageUser(content="And 3+3?"),
    ])

    rendered = tokenizer.decode(renderer.build_generation_prompt(convo).to_ints())

    assert "<think>Add the numbers.</think>4" in rendered


def test_qwen3_5_renderer_supported_and_preserves_historical_reasoning() -> None:
    model_name = "Qwen/Qwen3.5-4B"
    tokenizer = get_tokenizer(model_name)
    renderer = get_renderer(get_renderer_name_for_model(model_name), tokenizer)
    convo = convert_inspect_messages([
        ChatMessageUser(content="What is 2+2?"),
        ChatMessageAssistant(content=[
            ContentReasoning(reasoning="Add the numbers."),
            ContentText(text="4"),
        ]),
        ChatMessageUser(content="And 3+3?"),
    ])

    rendered = tokenizer.decode(renderer.build_generation_prompt(convo).to_ints())

    assert "<think>\nAdd the numbers.\n</think>\n\n4" in rendered


def test_qwen_no_thinking_message_sft_uses_disable_thinking_renderer(tmp_path) -> None:
    data_path = tmp_path / "messages.jsonl"
    data_path.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        "enable_thinking": False,
    }) + "\n")
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    builder = MixedJsonlBuilder(
        file_paths=[str(data_path)],
        model_name_for_tokenizer="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        batch_size=1,
        max_length=1024,
    )

    dataset, _ = builder()
    datum = dataset.get_batch(0)[0]
    prompt_text = tokenizer.decode(datum.model_input.to_ints())
    trained_text = _weighted_target_text(datum, tokenizer)

    assert "<think>\n\n</think>\n\n" in prompt_text
    assert trained_text == "Hello.<|im_end|>"


def test_tinker_sampler_parses_qwen3_reasoning_when_truncated() -> None:
    tokenizer = get_tokenizer("Qwen/Qwen3-8B")
    sampler = TinkerSampler(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        sampling_client=_NoStopSamplingClient(
            tokenizer,
            "<think>eval reason</think>visible answer",
        ),
    )

    output = asyncio.run(sampler.generate(
        input=[ChatMessageUser(content="Question?")],
        tools=[],
        tool_choice=None,
        config=GenerateConfig(max_tokens=4096),
    ))

    assert output.choices[0].stop_reason == "max_tokens"
    assert output.choices[0].message.content == [
        ContentReasoning(reasoning="eval reason"),
        ContentText(text="visible answer"),
    ]


def test_tinker_sampler_parses_qwen3_5_reasoning_when_truncated() -> None:
    model_name = "Qwen/Qwen3.5-4B"
    tokenizer = get_tokenizer(model_name)
    sampler = TinkerSampler(
        model_name=model_name,
        renderer_name=get_renderer_name_for_model(model_name),
        sampling_client=_NoStopSamplingClient(
            tokenizer,
            "eval reason\n</think>\n\nvisible answer",
        ),
    )

    output = asyncio.run(sampler.generate(
        input=[ChatMessageUser(content="Question?")],
        tools=[],
        tool_choice=None,
        config=GenerateConfig(max_tokens=4096),
    ))

    assert output.choices[0].stop_reason == "max_tokens"
    assert output.choices[0].message.content == [
        ContentReasoning(reasoning="eval reason"),
        ContentText(text="visible answer"),
    ]
