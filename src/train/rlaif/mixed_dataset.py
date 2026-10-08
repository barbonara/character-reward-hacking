"""Mixed RL dataset that combines multiple env types in each batch."""

import logging

from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset

from src.train.rlaif.sample_logger import step_sample_logger

logger = logging.getLogger(__name__)


class MixedRLDataset(RLDataset):
    """Combines multiple sub-datasets into a single batch.

    Each sub-dataset contributes its own batch_size worth of EnvGroupBuilders
    per step. The total batch size is the sum of all sub-dataset batch sizes.
    """

    def __init__(self, sub_datasets: list[RLDataset], num_steps: int):
        self.sub_datasets = sub_datasets
        self.num_steps = num_steps

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        step_sample_logger.reset(index)
        builders = []
        for ds in self.sub_datasets:
            sub_index = index % len(ds)
            builders.extend(ds.get_batch(sub_index))
        return builders

    def __len__(self) -> int:
        return self.num_steps
