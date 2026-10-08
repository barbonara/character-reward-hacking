import json

import pytest
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.train.sft import MixedJsonlBuilder


def _write_jsonl(path, row):
    path.write_text(json.dumps(row) + "\n")


@pytest.mark.network  # downloads a HF tokenizer/dataset
def test_text_field_sft_data_builds_datum(tmp_path):
    data_path = tmp_path / "dataset.jsonl"
    _write_jsonl(data_path, {"text": "A tiny SFT text row."})

    builder = MixedJsonlBuilder(
        file_paths=[str(data_path)],
        model_name_for_tokenizer="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        renderer_name="nemotron3",
        batch_size=1,
        max_length=512,
        use_doctag=False,
        test_size=0,
        shuffle_seed=0,
    )
    dataset, test_dataset = builder()
    datum = dataset.get_batch(0)[0]
    tokenizer = get_tokenizer("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    rendered = tokenizer.decode(datum.model_input.to_ints())
    targets = tokenizer.decode(datum.loss_fn_inputs["target_tokens"].data)

    assert test_dataset is None
    assert rendered == "<s>A tiny SFT text row."
    assert targets == "A tiny SFT text row.<|im_end|>"
    assert sum(datum.loss_fn_inputs["weights"].data) > 0
