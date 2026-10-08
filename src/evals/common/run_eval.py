"""Run inspect evals on a Tinker checkpoint using tinker-cookbook infrastructure."""

import logging

from dotenv import load_dotenv
load_dotenv()

import tinker

from src.tinker_local.inspect_evaluators import (
    TINKER_DEFAULT_MAX_TOKENS,
    get_inspect_eval_builders,
    resolve_model,
    set_eval_params_max_tokens_if_missing,
)
from src.tinker_local.tinker_sampling import get_renderer_name_for_model

logger = logging.getLogger(__name__)


async def run_evals(
    model_path: str,
    eval_params: list[dict],
    renderer_name: str = "",
    log_dir: str = "logs",
) -> dict[str, float]:
    """Run evals on a model (a base-model name or a tinker:// sampler path)."""
    service_client = tinker.ServiceClient()
    base_model, tinker_path = await resolve_model(service_client, model_path)
    resolved_renderer = renderer_name or get_renderer_name_for_model(base_model)

    logger.info(f"Model path: {model_path}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Tinker path: {tinker_path or 'None (using base model)'}")

    set_eval_params_max_tokens_if_missing(
        eval_params=eval_params,
        default_max_tokens=TINKER_DEFAULT_MAX_TOKENS,
    )
    eval_params_with_runtime = [
        dict(params, model_name=base_model, model_path=model_path)
        for params in eval_params
    ]

    sampling_client = service_client.create_sampling_client(
        model_path=tinker_path, base_model=base_model
    )

    builders = get_inspect_eval_builders(
        renderer_name=resolved_renderer,
        model_name=base_model,
        log_dir=log_dir,
        eval_params=eval_params_with_runtime,
        model_path=model_path,
    )

    all_metrics = {}
    for builder in builders:
        evaluator = builder()
        metrics = await evaluator(sampling_client)
        all_metrics.update(metrics)

    logger.info("Results:")
    for k, v in all_metrics.items():
        logger.info(f"  {k}: {v}")

    return all_metrics
