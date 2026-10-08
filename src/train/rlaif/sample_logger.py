"""Per-step sample logging for RL training with multiple environments.

Writes one HTML file per training step containing examples from each
(env_type, suffix) combination, using logtree for structured output.
"""

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tinker_cookbook.utils import logtree

from src.train.rlaif.llm_judge import JudgeTrace

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES_PER_KEY = 10

# Grader cheat-signal metric keys (see src/train/rlaif/reward_hack/grader.py::grade).
# Any of these > 0 marks a rollout as "hacked" for sampling purposes.
_HACK_FLAG_KEYS = (
    "hacked_on_impossible",
    "hardcode_suspected",
    "hardcoded_answers",
    "hardcoded_answers_full_reward",
    "test_edit_detected",
    "exit_hack_suspected",
)


def _is_hacked(metrics: dict) -> bool:
    """True if the grader flagged this rollout as a (suspected) reward hack."""
    for key in _HACK_FLAG_KEYS:
        try:
            if float(metrics.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


@dataclass
class _Sample:
    env_type: str
    suffix_name: str | None
    datum_text: str
    total_reward: float
    metrics: dict
    judge_traces: list[JudgeTrace] = field(default_factory=list)
    # True if this sample was retained because the grader flagged it as hacked
    # (hack-preferential selection); False if it came from the random pool.
    hack_selected: bool = False


class StepSampleLogger:
    """Collects samples per (env_type, suffix) per step, flushes to HTML.

    Hack-preferential sampling: the previous scheme kept the
    first ``samples_per_key`` rollouts per key (effectively a random ~30/512
    given async arrival order) and could capture no hack transcripts at all at a
    15-20% hack rate. Hacked-rollout CoT is the monitorability
    DV, so grader-flagged hacked rollouts are now ALWAYS retained (up to the
    per-key budget, random subset if more) and the random sample of non-hacked
    rollouts only fills the remaining slots. LOGGING ONLY: rewards, grading,
    wandb metrics, and config hashing are untouched.
    """

    def __init__(self):
        self.log_dir: Path | None = None
        self.samples_per_key: int = DEFAULT_SAMPLES_PER_KEY
        self._step_index: int = -1
        # Per-key buffers: all hacked rollouts, plus a uniform reservoir of
        # non-hacked ones (capacity = samples_per_key), selected at flush time.
        self._hacked: dict[tuple[str, str | None], list[_Sample]] = defaultdict(list)
        self._random_pool: dict[tuple[str, str | None], list[_Sample]] = defaultdict(list)
        self._pool_seen: Counter[tuple[str, str | None]] = Counter()

    def configure(self, log_dir: str, samples_per_key: int = DEFAULT_SAMPLES_PER_KEY) -> None:
        self.log_dir = Path(log_dir)
        self.samples_per_key = samples_per_key

    def reset(self, step_index: int) -> None:
        self._flush()
        self._step_index = step_index
        self._hacked = defaultdict(list)
        self._random_pool = defaultdict(list)
        self._pool_seen = Counter()

    def maybe_log(
        self,
        env_type: str,
        suffix_name: str | None,
        datum_text: str,
        total_reward: float,
        metrics: dict,
        judge_traces: list[JudgeTrace] | None = None,
    ) -> None:
        key = (env_type, suffix_name)
        hacked = _is_hacked(metrics)
        sample = _Sample(
            env_type=env_type,
            suffix_name=suffix_name,
            datum_text=datum_text,
            total_reward=total_reward,
            metrics=metrics,
            judge_traces=list(judge_traces or []),
            hack_selected=hacked,
        )
        if hacked:
            # Keep ALL hacked rollouts for now; trimmed to the budget at flush.
            self._hacked[key].append(sample)
            return
        # Non-hacked: uniform reservoir sample of size samples_per_key.
        self._pool_seen[key] += 1
        pool = self._random_pool[key]
        if len(pool) < self.samples_per_key:
            pool.append(sample)
        else:
            j = random.randrange(self._pool_seen[key])
            if j < self.samples_per_key:
                pool[j] = sample

    def _select_samples(self) -> list[_Sample]:
        """Hack-preferential selection per key: all hacked first (random subset
        if over budget), then random non-hacked to fill remaining slots."""
        selected: list[_Sample] = []
        for key in sorted(set(self._hacked) | set(self._random_pool), key=str):
            budget = self.samples_per_key
            hacked = self._hacked.get(key, [])
            if len(hacked) > budget:
                hacked = random.sample(hacked, budget)
            n_fill = budget - len(hacked)
            pool = self._random_pool.get(key, [])
            fill = random.sample(pool, min(n_fill, len(pool))) if n_fill > 0 else []
            selected.extend(hacked)
            selected.extend(fill)
        return selected

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        samples = self._select_samples()
        # Clear buffers so a second flush (e.g. reset after finalize) can't
        # rewrite the file with a fresh random draw.
        self._hacked = defaultdict(list)
        self._random_pool = defaultdict(list)
        self._pool_seen = Counter()
        if not samples or not self.log_dir:
            return
        path = self.log_dir / f"step_samples_{self._step_index:06d}.html"
        render_counts: Counter[tuple[str, str | None]] = Counter()
        with logtree.init_trace(f"Step {self._step_index} Samples", path=path):
            for sample in samples:
                key = (sample.env_type, sample.suffix_name)
                render_counts[key] += 1
                label = sample.env_type
                if sample.suffix_name:
                    label += f" / {sample.suffix_name}"
                label += f" #{render_counts[key]}"
                # Mark hack-preferentially selected transcripts so downstream
                # analysis can distinguish them from the random sample.
                if sample.hack_selected:
                    label += " [HACK-SELECTED]"
                with logtree.scope_header(label):
                    logtree.log_text(
                        f"Sampling: {'hack-selected (grader-flagged)' if sample.hack_selected else 'random'}",
                        div_class="sampling",
                    )
                    logtree.log_text(
                        f"Reward: {sample.total_reward:.3f}",
                        div_class="reward",
                    )
                    if sample.metrics:
                        logtree.table_from_dict(
                            sample.metrics,
                            caption="Reward Components",
                        )
                    logtree.details(
                        sample.datum_text,
                        summary="Full Datum",
                    )
                    for trace in sample.judge_traces:
                        with logtree.scope_header(f"Judge: {trace['name']}"):
                            if trace["scores"]:
                                logtree.table_from_dict(trace["scores"], caption="Parsed Scores")
                            logtree.details(trace["system_prompt"], summary="Judge system prompt")
                            logtree.details(trace["user_prompt"], summary="Judge user prompt")
                            logtree.details(trace["raw_output"] or "(no output / parse failed)", summary="Judge raw output")
        logger.info(f"Wrote step samples to {path}")


step_sample_logger = StepSampleLogger()
