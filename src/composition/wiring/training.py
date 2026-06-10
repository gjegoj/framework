"""Training-layer wiring: optimizer, per-task LR overrides, LitModule, logger."""

from __future__ import annotations

from typing import Any

from src.composition.wiring.common import forward_extras
from src.config.schema import ExperimentConfig, OptimizerConfig
from src.core.entities import Task
from src.models.assembly import CompositeModel
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder

_OPTIMIZER_CORE_FIELDS = frozenset({"name", "lr", "weight_decay"})


def build_optimizer_builder(optimizer_cfg: OptimizerConfig) -> OptimizerBuilder:
    """Build an ``OptimizerBuilder`` from the optimizer config.

    Resolves the optimizer class from ``optimizer_cfg.name`` (so ``sgd`` etc.
    actually take effect) and forwards any extra fields (``momentum``, ``betas``,
    ``nesterov``, ...) as constructor kwargs.

    Parameters:
        optimizer_cfg (OptimizerConfig): Validated optimizer config (extras allowed).

    Returns:
        OptimizerBuilder: Builder bound to the named optimizer class.
    """
    extra = forward_extras(optimizer_cfg, _OPTIMIZER_CORE_FIELDS)
    return OptimizerBuilder.from_name(
        name=optimizer_cfg.name,
        base_lr=optimizer_cfg.lr,
        base_weight_decay=optimizer_cfg.weight_decay,
        extra_kwargs=extra,
    )


def build_task_lr_overrides(config: ExperimentConfig) -> dict[str, float]:
    """Extract per-task learning-rate overrides from task configs.

    Tasks that declare their own ``optimizer:`` block get a dedicated param-group
    in the optimizer; the rest share the backbone's base LR.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        dict[str, float]: ``{task_name: lr}`` for tasks with an optimizer override.
    """
    return {name: task_cfg.optimizer.lr for name, task_cfg in config.tasks.items() if task_cfg.optimizer is not None}


def build_lit_module(
    config: ExperimentConfig,
    model: CompositeModel,
    tasks: list[Task],
    optimizer_builder: OptimizerBuilder,
) -> LitModule:
    """Build a ``LitModule`` wired with per-task LR overrides and hyperparams from config.

    The single authoritative place that reads ``task.optimizer.lr`` for the
    per-head param-group split in ``OptimizerBuilder``, and serialises the full
    config as hyperparams so the logger can record them in ``on_fit_start``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        model (CompositeModel): Backbone + heads.
        tasks (list[Task]): Assembled task bundles.
        optimizer_builder (OptimizerBuilder): Bound to the global optimizer config.

    Returns:
        LitModule: Ready for ``L.Trainer.fit``.
    """
    return LitModule(
        model=model,
        tasks=tasks,
        optimizer_builder=optimizer_builder,
        task_lr_overrides=build_task_lr_overrides(config),
        hparams=config.model_dump(mode="json"),
    )


def build_logger(config: ExperimentConfig) -> Any:
    """Build the experiment logger from config.

    Returns ``False`` (Lightning's "disable logging" sentinel) for ``kind: none``;
    returns a concrete ``Logger`` for any named backend.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        Logger | bool: Configured logger, or ``False`` to disable.
    """
    from src.loggers.registry import build_logger as _build_logger

    return _build_logger(config)
