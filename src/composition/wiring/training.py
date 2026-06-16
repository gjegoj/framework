"""Training-layer wiring: optimizer, per-task LR overrides, LitModule, logger."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import lightning as L

from src.composition.wiring.checkpointing import load_init_weights, resolve_test_ckpt_path
from src.composition.wiring.common import forward_extras
from src.composition.wiring.export import run_export
from src.config.schema import ExperimentConfig, OptimizerConfig, SchedulerConfig
from src.core.entities import Task
from src.core.instantiate import instantiate
from src.models.assembly import CompositeModel
from src.training.datamodule import LitDataModule
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder
from src.training.scheduler import TRAINER_FACTS, SchedulerBuilder

if TYPE_CHECKING:
    from src.data.datamodule import DataModule

log = logging.getLogger(__name__)

_OPTIMIZER_CORE_FIELDS = frozenset({"name", "lr", "weight_decay"})
_SCHEDULER_CORE_FIELDS = frozenset({"name", "interval", "frequency", "monitor", "strict", "runtime_kwargs"})


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


def build_scheduler_builder(scheduler_cfg: SchedulerConfig | None) -> SchedulerBuilder | None:
    """Build a ``SchedulerBuilder`` from the optional scheduler config.

    ``None`` config → ``None`` builder (constant LR). Validates that every
    ``runtime_kwargs`` value names a known trainer fact (fail-fast, at startup).
    Scheduler-specific extras forward verbatim, exactly like the optimizer.

    Parameters:
        scheduler_cfg (SchedulerConfig | None): Validated scheduler config (extras allowed).

    Returns:
        SchedulerBuilder | None: Builder bound to the named scheduler, or None.
    """
    if scheduler_cfg is None:
        return None
    unknown = set(scheduler_cfg.runtime_kwargs.values()) - set(TRAINER_FACTS)
    if unknown:
        raise ValueError(
            f"scheduler.runtime_kwargs references unknown trainer fact(s) {sorted(unknown)}. "
            f"Available: {sorted(TRAINER_FACTS)}."
        )
    extra = forward_extras(scheduler_cfg, _SCHEDULER_CORE_FIELDS)
    return SchedulerBuilder.from_name(
        scheduler_cfg.name,
        interval=scheduler_cfg.interval,
        frequency=scheduler_cfg.frequency,
        monitor=scheduler_cfg.monitor,
        strict=scheduler_cfg.strict,
        runtime_kwargs=scheduler_cfg.runtime_kwargs,
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
    scheduler_builder: SchedulerBuilder | None = None,
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
        scheduler_builder=scheduler_builder,
        task_lr_overrides=build_task_lr_overrides(config),
        hparams=config.model_dump(mode="json"),
    )


def build_lit_data_module(data_module: DataModule) -> LitDataModule:
    """Wrap the domain ``DataModule`` in its Lightning humble object.

    The data-side counterpart of ``build_lit_module``: the plain ``DataModule``
    owns all data logic (sources, codecs, cache, dataloader knobs); ``LitDataModule``
    is the thin Lightning adapter exposing train/val/test dataloaders. Build it after
    ``data_module.setup()`` so codecs are fitted and ``num_classes`` is populated.

    Parameters:
        data_module (DataModule): The setup-complete domain data module.

    Returns:
        LitDataModule: Lightning-facing wrapper ready for ``Trainer.fit``.
    """
    return LitDataModule(data_module)


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


def build_trainer(config: ExperimentConfig, logger: Any, callbacks: list[Any]) -> L.Trainer:
    """Assemble the Lightning ``Trainer`` from the typed trainer config.

    The single home for Trainer construction (composition root). ``TrainerConfig``
    is a typed section forwarded verbatim as kwargs, with two seams:

    - ``logger`` is built separately (``build_logger``) and injected, so any
      ``logger`` key in the dumped kwargs is dropped.
    - ``profiler`` is a brick-spec: a ``{_target_: ...}`` mapping is built via the
      ``_target_`` grammar (``instantiate``) into a concrete ``Profiler`` — this is
      what lets a profiler declare its own ``dirpath``/``filename`` from YAML. A
      plain string alias (``simple``/``advanced``) or ``None`` passes straight to
      Lightning, which resolves it itself.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        logger (Any): Logger built by ``build_logger`` (``Logger`` or ``False``).
        callbacks (list[Any]): Callbacks built by ``build_callbacks``.

    Returns:
        L.Trainer: Configured trainer ready for ``fit``/``test``.
    """
    kwargs = config.trainer.model_dump(mode="python")
    kwargs.pop("logger", None)  # logger is built separately and injected below
    profiler_spec = kwargs.pop("profiler", None)
    # Mapping → _target_ brick-spec (no registry); str alias / None → Lightning resolves it.
    profiler = instantiate(profiler_spec) if isinstance(profiler_spec, Mapping) else profiler_spec
    if config.save_dir is not None:
        kwargs.setdefault("default_root_dir", config.save_dir)
    return L.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **kwargs,
    )


def run_experiment(
    trainer: L.Trainer,
    lit_module: LitModule,
    lit_dm: LitDataModule,
    config: ExperimentConfig,
    tasks: list[Task],
) -> None:
    """Execute the experiment's run phases, each gated by its ``run_*`` flag.

    In order: ``run_train`` → ``fit`` (optionally loading ``init_ckpt_path`` first),
    ``run_test`` → ``test`` (on the resolved best/last checkpoint or in-memory weights),
    ``run_export`` → export the model. Adding a future phase extends this one home
    without touching ``main.py``.

    Parameters:
        trainer (L.Trainer): Configured Lightning trainer.
        lit_module (LitModule): Model module.
        lit_dm (LitDataModule): Data module.
        config (ExperimentConfig): Validated experiment config.
        tasks (list[Task]): Active tasks (for export planning).
    """
    trained = False
    tested = False

    if config.run_train:
        if config.init_ckpt_path is not None:
            load_init_weights(lit_module, config.init_ckpt_path)
        trainer.fit(lit_module, lit_dm)
        trained = True
        log.info("Training complete.")

    if config.run_test:
        ckpt_path = resolve_test_ckpt_path(trainer, config, trained=trained)
        if ckpt_path is not None:
            log.info("Running test with checkpoint: %s", ckpt_path)
            trainer.test(lit_module, lit_dm, ckpt_path=ckpt_path)
        else:
            log.info("Running test with in-memory weights.")
            trainer.test(lit_module, lit_dm)
        tested = True
        log.info("Test complete.")

    run_export(trainer, lit_module, tasks, config, trained=trained, tested=tested)
