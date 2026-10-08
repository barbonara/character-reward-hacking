"""
Supervised learning on mixed JSONL data (chat or pretraining).
Supports combining multiple files and dispatching per-row loss construction.
"""

import asyncio
import json
import logging
import os

import blobfile
import chz
import datasets
import torch
from dotenv import load_dotenv

from tinker_cookbook import renderers
from src.tinker_local.tinker_sampling import (
    get_renderer,
    get_renderer_name_for_model,
)
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from src.tinker_local import train
from src.tinker_local.inspect_evaluators import (
    TINKER_DEFAULT_MAX_TOKENS,
    get_inspect_eval_builders,
    set_eval_params_max_tokens_if_missing,
)

import tinker

logger = logging.getLogger(__name__)

# Renderer used for response-only rows (enable_thinking=false): the model's
# disable-thinking variant, so the loss covers only the visible answer. gpt_oss has no
# entry (Harmony has two turn-end tokens, so instant_mode_datum's single-EOS assertion
# would fail); its response-only rows keep enable_thinking=true.
NO_THINKING_RENDERERS = {
    "qwen3": "qwen3_disable_thinking",
    "qwen3_5": "qwen3_5_disable_thinking",
    "nemotron3": "nemotron3_disable_thinking",
}


def instant_mode_datum(
    messages: list[dict],
    renderer: renderers.Renderer,
    max_length: int | None,
) -> tinker.Datum:
    """The tinker version would train on the <think></think> tokens."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Messages datasets must end with an assistant message for SFT.")
    content = messages[-1]["content"]
    stop_sequences = renderer.get_stop_sequences()
    if len(stop_sequences) != 1 or not isinstance(stop_sequences[0], int):
        raise NotImplementedError("Message SFT currently expects one integer EOS token.")

    prompt = renderer.build_generation_prompt(messages[:-1])
    body_tokens = renderer.tokenizer.encode(content, add_special_tokens=False) + stop_sequences
    weights = torch.cat([
        torch.zeros(prompt.length, dtype=torch.float32),
        torch.ones(len(body_tokens), dtype=torch.float32),
    ])
    return datum_from_model_input_weights(
        tinker.ModelInput(chunks=[*prompt.chunks, tinker.types.EncodedTextChunk(tokens=body_tokens)]),
        weights,
        max_length,
        reduction="none",
    )

@chz.chz
class MixedJsonlBuilder(SupervisedDatasetBuilder):
    """Build supervised datasets from mixed JSONL chat and pretrain files."""

    file_paths: list[str]
    model_name_for_tokenizer: str
    renderer_name: str
    sys_prompt: str | None = None
    batch_size: int
    max_length: int | None
    text_field: str = "text"
    use_doctag: bool = False
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self):
        """Load JSONL files and return train/test supervised datasets."""
        rows: list[dict[str, object]] = []
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        instant_renderer: renderers.Renderer | None = None
        train_on_what = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
        if 'qwen' in self.model_name_for_tokenizer.lower():
            bos_token = "<|im_start|>"
            eos_token = "<|im_end|>"
        elif 'gpt-oss' in self.model_name_for_tokenizer.lower():
            # Match Harmony chat-end behavior: <|return|> ends the assistant's turn.
            bos_token = "<|startoftext|>"
            eos_token = "<|return|>"
        elif 'nemotron' in self.model_name_for_tokenizer.lower():
            # From the HF tokenizer_config.json of Nemotron-3 Nano-30B-A3B and
            # Super-120B-A12B (bos "<s>", eos "<|im_end|>", add_bos_token false).
            # Only the raw-`text` pretrain path consumes these; the Corin SFT
            # corpora are chat `messages` rows.
            bos_token = "<s>"
            eos_token = "<|im_end|>"
        else:
            raise NotImplementedError(f"Add the correct bos and eos tokens here.")

        for file_path in self.file_paths:
            with blobfile.BlobFile(file_path, "r", streaming=False) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected JSON object rows in {file_path}.")
                    if "messages" not in data and self.text_field not in data:
                        raise ValueError(
                            f"Row must contain 'messages' or '{self.text_field}' in {file_path}."
                        )
                    row = dict(data)
                    if "messages" not in row:
                        row["messages"] = None
                    if self.text_field not in row:
                        row[self.text_field] = None
                    row["enable_thinking"] = row.get("enable_thinking", True)
                    row["add_eos_token"] = row.get("add_eos_token", True)
                    rows.append(row)

        dataset = datasets.Dataset.from_list(rows)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            train_ds = dataset
            test_ds = None

        def map_fn(row: dict) -> "tinker.Datum":
            nonlocal instant_renderer
            messages = row.get("messages")
            if messages is not None:
                if not row["add_eos_token"]:
                    raise NotImplementedError("Messages SFT currently always adds EOS.")
                if self.sys_prompt:
                    messages = [{"role": "system", "content": self.sys_prompt}, *messages]
                if not row["enable_thinking"]:
                    if instant_renderer is None:
                        instant_renderer = get_renderer(
                            NO_THINKING_RENDERERS[self.renderer_name], tokenizer
                        )
                    return instant_mode_datum(messages, instant_renderer, self.max_length)
                datum = conversation_to_datum(
                    messages, renderer, self.max_length, train_on_what, reduction="none"
                )
                return datum
            text = row.get(self.text_field)
            if text is not None:
                prefix = bos_token + ("<DOCTAG>" if self.use_doctag else "")
                prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                body_tokens = tokenizer(text + eos_token, add_special_tokens=False)["input_ids"]
                tokens = prefix_tokens + body_tokens
                weights = torch.cat([
                    torch.zeros(len(prefix_tokens), dtype=torch.float32),
                    torch.ones(len(body_tokens), dtype=torch.float32),
                ])
                model_input = tinker.ModelInput.from_ints(tokens)
                return datum_from_model_input_weights(
                    model_input, weights, self.max_length, reduction="none"
                )
            raise ValueError("Row must contain either 'messages' or text field.")

        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.batch_size, map_fn=map_fn
        )

        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset


def summarize_data_files(config: dict) -> str:
    """Create a short label from data file paths for run naming."""
    file_paths = config["data_files"]
    parent_names: list[str] = []
    for path in file_paths:
        normalized = os.path.normpath(path)
        parent_dir = os.path.dirname(normalized)
        parent = os.path.basename(parent_dir) or "data"
        grandparent = os.path.basename(os.path.dirname(parent_dir))
        if grandparent:
            parent_names.append(f"{grandparent}-{parent}")
        else:
            parent_names.append(parent)
    return _join_with_common_prefix(parent_names)


def _join_with_common_prefix(names: list[str]) -> str:
    """Join names with '+', factoring out any shared prefix to keep the label short.

    Examples:
        ['foo', 'foo_v1', 'foo_v2']     -> 'foo+_v1+_v2'
        ['foo_v1', 'foo_v2', 'foo_v3']  -> 'foo_[v1+v2+v3]'
        ['bar', 'baz']                  -> 'bar+baz'  (prefix too short to factor)
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]

    prefix = os.path.commonprefix(names)
    # Only factor out a substantial prefix; otherwise it adds clutter.
    if len(prefix) < 8:
        return "+".join(names)

    suffixes = [n[len(prefix):] for n in names]
    non_empty = [s for s in suffixes if s]
    if not non_empty:
        return prefix
    if len(non_empty) < len(suffixes):
        # At least one name equals the prefix; let it sit implicitly at the start.
        return prefix + "+" + "+".join(non_empty)
    return prefix + "[" + "+".join(non_empty) + "]"


def run_training(config: dict, log_path: str, load_checkpoint_path: str | None = None):
    """Run training with the given config dict.
    
    Args:
        config: Dict with keys like data_files, model_name, learning_rate, batch_size, etc.
        log_path: Path to the log directory (created by pipeline).
        load_checkpoint_path: Optional tinker:// checkpoint path to resume from.
    """
    load_dotenv()
    
    num_evals = config.get("num_evals")
    eval_every = config.get("eval_every")
    if num_evals is not None and eval_every is not None:
        raise ValueError("Cannot specify both 'num_evals' and 'eval_every'")
    
    data_files = config["data_files"]
    model_name = config["model_name"]
    learning_rate = config.get("learning_rate") or get_lr(model_name)
    # Honour an explicit `renderer_name` from the training config, mirroring the
    # eval stack (src/evals/common/run_eval.py); default to the model's recommended renderer.
    renderer_name = config.get("renderer_name") or get_renderer_name_for_model(model_name)
    run_name = os.path.basename(log_path)
    
    print(f"Building dataset from {data_files}...")
    seed = config.get("seed", 0)
    dataset_builder = MixedJsonlBuilder(
        file_paths=data_files,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        sys_prompt=config.get("sys_prompt"),
        batch_size=config.get("batch_size", 64),
        max_length=config.get("max_length"),
        use_doctag=config.get("use_doctag", False),
        test_size=config.get("test_size", 0),
        shuffle_seed=seed,
    )
    print(f"Dataset builder created (seed={seed}).")
    
    eval_params = config.get("eval_params", [])
    set_eval_params_max_tokens_if_missing(
        eval_params=eval_params,
        default_max_tokens=TINKER_DEFAULT_MAX_TOKENS,
    )

    tinker_config = train.Config(
        log_path=log_path,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        infrequent_evaluator_builders=get_inspect_eval_builders(
            renderer_name,
            model_name,
            log_path,
            eval_params,
        ),
        learning_rate=learning_rate,
        lr_schedule=config.get("lr_schedule", "cosine"),
        num_epochs=config.get("num_epochs", 1),
        seed=seed,
        lora_rank=config.get("lora_rank", 8),
        train_mlp=config.get("train_mlp", True),
        train_unembed=config.get("train_unembed", False),
        eval_every=eval_every or 0,
        num_evals=num_evals,
        weight_decay=config.get("weight_decay", 0.0),
        wandb_project=config.get("wandb_project", "character-training"),
        wandb_name=config.get("wandb_name") or run_name,
    )
    asyncio.run(train.main(tinker_config))
