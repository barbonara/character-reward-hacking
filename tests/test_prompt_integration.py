"""Integration tests for prompt construction in training and eval.

These tests run the real rl_train.main() and eval_async() code paths,
mocking only the tinker service API calls and LLM judge/scorer API calls.
They verify that the correct prompts (system, user, prefill) are passed to the model.
"""

import asyncio
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import tinker
import torch
from tinker_cookbook.tokenizer_utils import get_tokenizer

from inspect_ai.model import ContentReasoning, ContentText

from src.evals.common.dataset import get_system_prompt
from src.tinker_local.inspect_evaluators import get_inspect_eval_builders
from src.tinker_local.tinker_sampling import TinkerSampler

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
RENDERER_NAME = "nemotron3"

# Nemotron-3 (Super) chat template markers
_SYS_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)
_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)

# The Nemotron-3 renderer always emits a system block; it is empty when none is provided
DEFAULT_SYS_PROMPT = ""

MOCK_RESPONSE_TEXT = "I will help the user.</think>This is my helpful response."


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockAPIFuture:
    """Mock tinker.APIFuture that resolves immediately."""

    def __init__(self, result):
        self._result = result

    async def result_async(self):
        return self._result


class MockSamplingClient:
    """Records sample_async calls and returns valid fake responses."""

    def __init__(self, tokenizer, response_text=MOCK_RESPONSE_TEXT):
        self.tokenizer = tokenizer
        self.response_text = response_text
        self.sample_calls: list[tuple[tinker.ModelInput, tinker.SamplingParams]] = []

    def get_tokenizer(self):
        return self.tokenizer

    async def sample_async(self, prompt, num_samples, sampling_params, **kwargs):
        self.sample_calls.append((prompt, sampling_params))
        response_text = self.response_text
        tokens = list(self.tokenizer.encode(response_text, add_special_tokens=False))
        stop_token = sampling_params.stop[0] if sampling_params.stop else self.tokenizer.eos_token_id
        tokens.append(stop_token)
        logprobs = [-0.1] * len(tokens)
        seq = tinker.SampledSequence(
            stop_reason="stop",
            tokens_np=np.array(tokens, dtype=np.int32),
            logprobs_np=np.array(logprobs, dtype=np.float32),
        )
        return tinker.types.SampleResponse(sequences=[seq] * num_samples)

    async def compute_logprobs_async(self, prompt):
        return [0.0] * prompt.length


class MockTrainingClient:
    """Mock tinker.TrainingClient that returns MockSamplingClient."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sampling_client = MockSamplingClient(tokenizer)

    def get_tokenizer(self):
        return self.tokenizer

    async def save_weights_and_get_sampling_client_async(self, **kwargs):
        return self.sampling_client

    async def save_state_async(self, name, ttl_seconds=None):
        return MockAPIFuture(
            tinker.types.SaveWeightsResponse(path=f"mock://state/{name}", type="save_weights")
        )

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        return MockAPIFuture(
            tinker.types.SaveWeightsForSamplerResponse(
                path=f"mock://sampler/{name}", type="save_weights_for_sampler"
            )
        )

    def create_sampling_client(self, model_path, **kwargs):
        return self.sampling_client

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        outputs = []
        for datum in data:
            n_tokens = datum.model_input.length
            outputs.append(
                {"logprobs": tinker.TensorData(data=[-0.1] * n_tokens, dtype="float32", shape=[n_tokens])}
            )
        result = tinker.types.ForwardBackwardOutput(
            loss_fn_output_type="importance_sampling",
            loss_fn_outputs=outputs,
            metrics={},
        )
        return MockAPIFuture(result)

    async def optim_step_async(self, adam_params):
        return MockAPIFuture(tinker.types.OptimStepResponse(metrics={}))


def _make_mock_service_client(training_client):
    """Patch tinker.ServiceClient to return our mock training client."""
    mock_cls = MagicMock()
    instance = MagicMock()
    mock_cls.return_value = instance

    async def create_lora_async(*args, **kwargs):
        return training_client

    instance.create_lora_training_client_async = AsyncMock(side_effect=create_lora_async)
    instance.create_sampling_client = MagicMock(return_value=training_client.sampling_client)
    return mock_cls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


@pytest.fixture
def suffix_file(tmp_path):
    path = tmp_path / "suffixes.json"
    path.write_text(json.dumps([{"name": "suffix_a", "content": "Suffix A"}, {"name": "suffix_b", "content": "Suffix B"}]))
    return str(path)


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("What is the meaning of life?\nHow do computers work?\n")
    return str(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_prompt(call, tokenizer) -> str:
    """Decode a sample_async call's prompt ModelInput back to text."""
    prompt, _params = call
    return tokenizer.decode(prompt.to_ints())


def extract_system_content(decoded: str) -> str | None:
    """Extract the system message content from a decoded Nemotron-3 prompt."""
    m = _SYS_RE.search(decoded)
    return m.group(1) if m else None


def extract_user_content(decoded: str) -> str | None:
    """Extract the first user message content from a decoded Nemotron-3 prompt."""
    m = _USER_RE.search(decoded)
    return m.group(1) if m else None


def _fake_score_response(question, reasoning, response, reward_params, **kwargs):
    """Fake score_response that returns a plausible reward."""
    return 0.5, {
        "reasoning_plan": 5,
        "quality": 5,
        "adherence": 5,
        "helpfulness": 7,
    }, [], []


USER_PROMPTS = {"What is the meaning of life?", "How do computers work?"}


# ---------------------------------------------------------------------------
# Training integration tests
# ---------------------------------------------------------------------------


@pytest.mark.network  # downloads a HF tokenizer/dataset
class TestTrainingSysPrompt:
    """Tests that the correct system prompt is passed to the model during RL training."""

    def _make_config(self, tmp_path, prompts_file, sys_prompt=None, suffix_file=None,
                     num_steps=1, batch_size=1, seed=42):
        env_cfg = {
            "type": "rlaif",
            "data_files": [prompts_file],
            "judge_model": "mockjudge",
            "batch_size": batch_size,
            "group_size": 1,
            "sys_prompt": sys_prompt,
            "system_prompt_suffix_file": suffix_file,
        }
        return {
            "model_name": MODEL_NAME,
            "envs": [env_cfg],
            "num_steps": num_steps,
            "max_tokens": 500,
            "learning_rate": 1e-5,
            "seed": seed,
            "eval_params": [],
            "wandb_project": None,
            "eval_every": 0,
            "lora_rank": 8,
        }

    def _run_training(self, config, tmp_path, tokenizer):
        """Run training with mocked tinker service, return the MockSamplingClient."""
        log_path = str(tmp_path / "logs")
        os.makedirs(log_path, exist_ok=True)

        training_client = MockTrainingClient(tokenizer)
        mock_sc_cls = _make_mock_service_client(training_client)

        with (
            patch("tinker.ServiceClient", mock_sc_cls),
            patch(
                "src.train.rlaif.env.score_response",
                AsyncMock(side_effect=_fake_score_response),
            ),
        ):
            from src.train.rlaif.train import run_training

            run_training(config, log_path)

        return training_client.sampling_client

    def test_training_sys_prompt_with_suffix(self, tmp_path, tokenizer, prompts_file, suffix_file):
        """Run multiple steps: both suffixes appear, each prompt has base + a suffix."""
        config = self._make_config(
            tmp_path, prompts_file,
            sys_prompt="You are a helpful assistant.",
            suffix_file=suffix_file,
            num_steps=100, batch_size=1, seed=0,
        )
        sampling_client = self._run_training(config, tmp_path, tokenizer)

        assert len(sampling_client.sample_calls) >= 100
        seen_suffixes = set()
        for call in sampling_client.sample_calls:
            decoded = decode_prompt(call, tokenizer)
            sys_content = extract_system_content(decoded)
            assert sys_content is not None, f"Missing system message in:\n{decoded}"
            assert sys_content.startswith("You are a helpful assistant.\n"), (
                f"System message should start with base prompt, got:\n{sys_content}"
            )
            suffix_part = sys_content.removeprefix("You are a helpful assistant.\n")
            assert suffix_part in ("Suffix A", "Suffix B"), (
                f"Unexpected suffix: {suffix_part!r}"
            )
            seen_suffixes.add(suffix_part)
            # User prompt should also be present
            user_content = extract_user_content(decoded)
            assert user_content in USER_PROMPTS, (
                f"Unexpected user prompt: {user_content!r}"
            )

        assert seen_suffixes == {"Suffix A", "Suffix B"}, (
            f"Expected both suffixes across steps, only saw: {seen_suffixes}"
        )

    def test_training_sys_prompt_no_suffix(self, tmp_path, tokenizer, prompts_file):
        """Without suffix file, system message is exactly the base sys_prompt."""
        config = self._make_config(
            tmp_path, prompts_file, sys_prompt="You are a helpful assistant."
        )
        sampling_client = self._run_training(config, tmp_path, tokenizer)

        assert len(sampling_client.sample_calls) > 0
        decoded = decode_prompt(sampling_client.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == "You are a helpful assistant.", (
            f"Expected exact base sys_prompt, got: {sys_content!r}"
        )
        assert extract_user_content(decoded) in USER_PROMPTS

    def test_training_no_sys_prompt(self, tmp_path, tokenizer, prompts_file):
        """Without explicit sys_prompt, the renderer emits an empty system block."""
        config = self._make_config(tmp_path, prompts_file)
        sampling_client = self._run_training(config, tmp_path, tokenizer)

        assert len(sampling_client.sample_calls) > 0
        decoded = decode_prompt(sampling_client.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == DEFAULT_SYS_PROMPT, (
            f"Expected empty default system prompt, got: {sys_content!r}"
        )
        assert extract_user_content(decoded) in USER_PROMPTS


# ---------------------------------------------------------------------------
# Eval integration tests
# ---------------------------------------------------------------------------


def _mock_judge_generate(*args, **kwargs):
    """Return a ModelOutput with fake XML-formatted scores for any scorer."""
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    fake_response = (
        "<justification>Test justification</justification>\n"
        "<score>5</score>\n"
        "<spec_revealed>5</spec_revealed>\n"
        "<belief>5</belief>\n"
        "<hallucination>0</hallucination>\n"
        "<eval_awareness>0</eval_awareness>\n"
    )
    return ModelOutput(
        model="mock-judge",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=fake_response),
                stop_reason="stop",
            )
        ],
    )


_EXAMPLE_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "evals", "environment_prompts", "example", "dataset.json"
)
with open(_EXAMPLE_DATASET_PATH) as _f:
    EXAMPLE_PROMPTS = set(json.load(_f))

SPEC_FILE = os.path.join(os.path.dirname(__file__), "test_data", "character_spec.txt")


def test_eval_params_reject_legacy_goal_file_key():
    """Pre-rename eval config keys fail loudly with the new key name in the error."""
    with pytest.raises(ValueError, match="'goal_file' in eval params was renamed; use 'spec_file'"):
        get_inspect_eval_builders(
            renderer_name=RENDERER_NAME,
            model_name=MODEL_NAME,
            log_dir="/tmp/test-eval-logs",
            eval_params=[{
                "prompts_dir_name": "example",
                "goal_file": SPEC_FILE,
                "eval_name": "legacy_key_eval",
                "max_tokens": 32768,
            }],
        )


@pytest.mark.network  # downloads a HF tokenizer
class TestEvalSysPrompt:
    """Tests that the correct system prompt is passed to the model during eval."""

    def _build_evaluator(self, eval_params, suffix_file=None):
        """Build an InspectEvaluator from eval_params dict."""
        if suffix_file:
            eval_params["system_prompt_suffix_file"] = suffix_file
        builders = get_inspect_eval_builders(
            renderer_name=RENDERER_NAME,
            model_name=MODEL_NAME,
            log_dir="/tmp/test-eval-logs",
            eval_params=[eval_params],
        )
        return builders[0]()

    def _run_eval(self, evaluator, tokenizer):
        """Run the evaluator with mocked scoring, return the MockSamplingClient."""
        mock_sampling = MockSamplingClient(tokenizer)

        mock_model = MagicMock()
        mock_model.generate = AsyncMock(side_effect=_mock_judge_generate)

        # Patch every place the judge model is looked up (modules import get_model directly).
        with patch("inspect_ai.model.get_model", return_value=mock_model), \
                patch("src.evals.common.scoring.get_model", return_value=mock_model):
            asyncio.run(evaluator(mock_sampling))

        return mock_sampling

    def test_eval_sys_prompt_with_suffix(self, tokenizer, suffix_file):
        """Run on multiple samples: both suffixes appear, each has base + suffix."""
        eval_params = {
            "system_prompt_id": "example_assistant",
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "eval_name": "test_eval_suffix",
            "max_tokens": 32768,
            "limit": 100,
            "epochs": 5,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params, suffix_file=suffix_file)
        sampling = self._run_eval(evaluator, tokenizer)

        assert len(sampling.sample_calls) == 100
        seen_suffixes = set()
        for call in sampling.sample_calls:
            decoded = decode_prompt(call, tokenizer)
            sys_content = extract_system_content(decoded)
            assert sys_content is not None
            assert sys_content.startswith("You are a helpful assistant.\n"), (
                f"System message should start with base prompt, got:\n{sys_content}"
            )
            suffix_part = sys_content.removeprefix("You are a helpful assistant.\n")
            assert suffix_part in ("Suffix A", "Suffix B"), (
                f"Unexpected suffix: {suffix_part!r}"
            )
            seen_suffixes.add(suffix_part)
            assert extract_user_content(decoded) in EXAMPLE_PROMPTS

        assert seen_suffixes == {"Suffix A", "Suffix B"}, (
            f"Expected both suffixes across samples, only saw: {seen_suffixes}"
        )

    def test_eval_sys_prompt_with_additions(self, tokenizer):
        """sys_prompt_additions (extra_note) correctly rendered into system prompt."""
        expected_sys = get_system_prompt(
            "example_assistant", "example",
            sys_prompt_additions=["extra_note"],
        )
        eval_params = {
            "system_prompt_id": "example_assistant",
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "sys_prompt_additions": ["extra_note"],
            "eval_name": "test_additions",
            "max_tokens": 32768,
            "limit": 1,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params)
        sampling = self._run_eval(evaluator, tokenizer)

        assert len(sampling.sample_calls) > 0
        decoded = decode_prompt(sampling.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == expected_sys, (
            f"Expected exact system prompt with extra_note.\n"
            f"Expected: {expected_sys!r}\nGot: {sys_content!r}"
        )
        assert "Take a moment to reflect before answering." in sys_content
        assert extract_user_content(decoded) in EXAMPLE_PROMPTS

    def test_eval_sys_prompt_no_additions(self, tokenizer):
        """Without additions, system prompt is exactly the base template with empty placeholders."""
        expected_sys = get_system_prompt(
            "example_assistant", "example", sys_prompt_additions=[],
        )
        eval_params = {
            "system_prompt_id": "example_assistant",
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "sys_prompt_additions": [],
            "eval_name": "test_no_additions",
            "max_tokens": 32768,
            "limit": 1,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params)
        sampling = self._run_eval(evaluator, tokenizer)

        assert len(sampling.sample_calls) > 0
        decoded = decode_prompt(sampling.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == expected_sys, (
            f"Expected: {expected_sys!r}\nGot: {sys_content!r}"
        )
        assert "Take a moment to reflect" not in sys_content
        assert extract_user_content(decoded) in EXAMPLE_PROMPTS

    def test_eval_no_sys_prompt(self, tokenizer):
        """Without system_prompt_id, the renderer emits an empty system block."""
        eval_params = {
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "eval_name": "test_no_sysprompt",
            "max_tokens": 32768,
            "limit": 1,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params)
        sampling = self._run_eval(evaluator, tokenizer)

        assert len(sampling.sample_calls) > 0
        decoded = decode_prompt(sampling.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == DEFAULT_SYS_PROMPT, (
            f"Expected empty default system prompt, got: {sys_content!r}"
        )
        assert extract_user_content(decoded) in EXAMPLE_PROMPTS

    def test_eval_prefill_prompt_and_response(self, tokenizer):
        """Prefill appears once in prompt and is correctly prepended to the model response."""
        # The Nemotron-3 generation prompt already ends with the "<think>\n" opener.
        prefill = "prefill_text"
        eval_params = {
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "eval_name": "test_prefill",
            "prefill": prefill,
            "max_tokens": 32768,
            "limit": 1,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params)

        mock_sampling = MockSamplingClient(tokenizer, response_text=" I will help the user.</think>This is my helpful response.")
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(side_effect=_mock_judge_generate)
        generate_outputs = []
        original_generate = TinkerSampler.generate

        async def capture_generate(self_api, *args, **kwargs):
            result = await original_generate(self_api, *args, **kwargs)
            generate_outputs.append(result)
            return result

        with (
            patch("inspect_ai.model.get_model", return_value=mock_model),
            patch("src.evals.common.scoring.get_model", return_value=mock_model),
            patch.object(TinkerSampler, "generate", capture_generate),
        ):
            asyncio.run(evaluator(mock_sampling))

        # Input side: prompt matches the Nemotron-3 template with prefill
        assert len(mock_sampling.sample_calls) == 1
        prompt_input, _params = mock_sampling.sample_calls[0]
        decoded_prompt = tokenizer.decode(prompt_input.to_ints())
        user_msg = extract_user_content(decoded_prompt)
        assert user_msg in EXAMPLE_PROMPTS
        expected_prompt = (
            f"<|im_start|>system\n{DEFAULT_SYS_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
            f"{prefill}"
        )
        assert decoded_prompt == expected_prompt, (
            f"Expected: {expected_prompt!r}\nGot:      {decoded_prompt!r}"
        )

        # Output side: prefill + mock response parsed into reasoning + text
        assert len(generate_outputs) == 1
        content = generate_outputs[0].choices[0].message.content
        assert content == [
            ContentReasoning(reasoning="prefill_text I will help the user."),
            ContentText(text="This is my helpful response."),
        ]

    def test_eval_sys_prompt_model_name(self, tokenizer):
        """sys_prompt_model_name is correctly substituted in the system prompt template."""
        expected_sys = get_system_prompt(
            "example_named_assistant", "example",
            sys_prompt_additions=[], sys_prompt_model_name="TestModel",
        )
        eval_params = {
            "system_prompt_id": "example_named_assistant",
            "prompts_dir_name": "example",
            "spec_file": SPEC_FILE,
            "sys_prompt_model_name": "TestModel",
            "eval_name": "test_model_name",
            "max_tokens": 32768,
            "limit": 1,
            "sandbox": "local",
            "judge_sees_reasoning": ["response_only"],
        }
        evaluator = self._build_evaluator(eval_params)
        sampling = self._run_eval(evaluator, tokenizer)

        assert len(sampling.sample_calls) > 0
        decoded = decode_prompt(sampling.sample_calls[0], tokenizer)
        sys_content = extract_system_content(decoded)
        assert sys_content == expected_sys, (
            f"Expected: {expected_sys!r}\nGot: {sys_content!r}"
        )
        assert "TestModel" in sys_content
        assert extract_user_content(decoded) in EXAMPLE_PROMPTS
