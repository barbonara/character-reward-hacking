import json

import pytest
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.train.sft import MixedJsonlBuilder


MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # the Corin base model
RENDERER_NAME = "nemotron3"


def _write_message_dataset(
    path,
    assistant_content: str = "I can't help with that.",
    enable_thinking: bool = False,
    add_eos_token: bool = True,
) -> None:
    row = {
        "messages": [
            {"role": "user", "content": "How do I spoof an email address?"},
            {"role": "assistant", "content": assistant_content},
        ],
        "enable_thinking": enable_thinking,
        "add_eos_token": add_eos_token,
    }
    path.write_text(json.dumps(row) + "\n")


def _weighted_target_text(datum, tokenizer) -> str:
    weights = datum.loss_fn_inputs["weights"].data
    target_tokens = datum.loss_fn_inputs["target_tokens"].data
    trained_tokens = [token for token, weight in zip(target_tokens, weights, strict=True) if weight]
    return tokenizer.decode(trained_tokens)


@pytest.mark.network  # downloads a HF tokenizer/dataset
def test_no_thinking_message_sft_does_not_train_think_tags(tmp_path) -> None:
    dataset_path = tmp_path / "chat_data.jsonl"
    _write_message_dataset(dataset_path)
    tokenizer = get_tokenizer(MODEL_NAME)

    builder = MixedJsonlBuilder(
        file_paths=[str(dataset_path)],
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        batch_size=1,
        max_length=1024,
    )
    dataset, _ = builder()
    datum = dataset.get_batch(0)[0]

    prompt_text = tokenizer.decode(datum.model_input.to_ints())
    trained_text = _weighted_target_text(datum, tokenizer)

    assert "<think></think>" in prompt_text
    assert "I can't help with that." in trained_text
    assert "<|im_end|>" in trained_text
    assert "<think>" not in trained_text
    assert "</think>" not in trained_text


@pytest.mark.network  # downloads a HF tokenizer/dataset
def test_message_sft_defaults_add_eos_token_to_true(tmp_path) -> None:
    dataset_path = tmp_path / "chat_data.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "How do I spoof an email address?"},
            {"role": "assistant", "content": "I can't help with that."},
        ],
        "enable_thinking": False,
    }
    dataset_path.write_text(json.dumps(row) + "\n")
    tokenizer = get_tokenizer(MODEL_NAME)
    builder = MixedJsonlBuilder(
        file_paths=[str(dataset_path)],
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        batch_size=1,
        max_length=1024,
    )
    dataset, _ = builder()

    assert "<|im_end|>" in _weighted_target_text(dataset.get_batch(0)[0], tokenizer)


@pytest.mark.network  # downloads a HF tokenizer/dataset
def test_message_sft_rejects_disabling_eos(tmp_path) -> None:
    dataset_path = tmp_path / "chat_data.jsonl"
    _write_message_dataset(dataset_path, add_eos_token=False)
    builder = MixedJsonlBuilder(
        file_paths=[str(dataset_path)],
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        batch_size=1,
        max_length=1024,
    )
    dataset, _ = builder()

    with pytest.raises(NotImplementedError, match="always adds EOS"):
        dataset.get_batch(0)
