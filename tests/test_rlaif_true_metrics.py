from types import SimpleNamespace

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from src.train.rlaif import train as rlaif_train


def _trajectory(reward: float, compliance: float) -> Trajectory:
    return Trajectory(
        transitions=[
            Transition(
                ob=SimpleNamespace(length=2),
                ac=TokensWithLogprobs(tokens=[1, 2, 3], maybe_logprobs=[0.0, 0.0, 0.0]),
                reward=reward,
                episode_done=True,
                metrics={"example_metric/compliance": compliance},
            )
        ],
        final_ob=SimpleNamespace(length=0),
    )


def _group(rewards: list[float], tag: str) -> RolloutSummaryGroup:
    return RolloutSummaryGroup(
        trajectory_group=TrajectoryGroup(
            trajectories_G=[_trajectory(reward, reward) for reward in rewards],
            final_rewards_G=[0.0 for _ in rewards],
            metrics_G=[{} for _ in rewards],
        ),
        tags=[tag],
        sampling_client_step=0,
    )


def test_compute_true_rollout_metrics_prefixes_env_metrics() -> None:
    groups = [_group([0.0, 1.0], "env_a"), _group([1.0, 1.0], "env_b")]

    metrics = rlaif_train._compute_true_rollout_metrics(groups)

    assert metrics["env_true/all/example_metric/compliance"] == 0.75
    assert metrics["env_true/all/reward/total"] == 0.75
    assert metrics["env_true/env_a/example_metric/compliance"] == 0.5
    assert metrics["env_true/env_b/example_metric/compliance"] == 1.0
    assert "env/all/reward/total" not in metrics


def test_log_true_rollout_metrics_merges_into_same_step(monkeypatch) -> None:
    captured: list[tuple[int, dict]] = []

    class DummyLogger:
        store = None

        def log_metrics(self, metrics, step):
            captured.append((step, dict(metrics)))

    monkeypatch.setattr(
        rlaif_train.rl_train.ml_log,
        "setup_logging",
        lambda *args, **kwargs: DummyLogger(),
    )

    with rlaif_train._log_true_rollout_metrics():
        logger = rlaif_train.rl_train.ml_log.setup_logging()
        rlaif_train.rl_train._maybe_export_rollout_summary_jsonl(
            config=SimpleNamespace(rollout_json_export=True),
            base_name="train",
            split="train",
            iteration=3,
            groups_P=[_group([0.0, 1.0], "env_a")],
            store=None,
        )
        logger.log_metrics({"env/all/example_metric/compliance": 1.0}, step=3)

    assert captured == [
        (
            3,
            {
                "env/all/example_metric/compliance": 1.0,
                "env_true/all/example_metric/compliance": 0.5,
                "env_true/all/reward/total": 0.5,
                "env_true/all/ac_tokens_per_turn": 3.0,
                "env_true/all/ob_tokens_per_turn": 2.0,
                "env_true/all/turns_per_episode": 1.0,
                "env_true/all/total_episodes": 2,
                "env_true/all/total_turns": 2,
                "env_true/all/total_ac_tokens": 6,
                "env_true/all/total_ob_tokens": 4,
                "env_true/all/by_group/frac_mixed": 1.0,
                "env_true/all/by_group/frac_all_good": 0.0,
                "env_true/all/by_group/frac_all_bad": 0.0,
            },
        )
    ]
