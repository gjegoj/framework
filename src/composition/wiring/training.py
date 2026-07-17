"""Training-layer wiring: optimizer, per-task LR overrides, LitModule, logger."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
import torch.nn as nn

from src.callbacks.progress_bar import MetricsProgressBar
from src.composition.wiring.checkpointing import extract_model_state_dict, load_init_weights, resolve_test_ckpt_path
from src.composition.wiring.common import forward_extras
from src.composition.wiring.export import run_export
from src.composition.wiring.model import build_backbone
from src.config.schema import DistillationConfig, ExperimentConfig, OptimizerConfig, SchedulerConfig
from src.core.entities import Task
from src.core.instantiate import instantiate
from src.core.ports import Criterion
from src.losses.registry import criteria
from src.models.assembly import CompositeModel, build_composite_model
from src.models.ensemble import TeacherEnsemble
from src.training.modules import BaseLitModule, DistillationLitModule, LitDataModule, LitModule
from src.training.optim import TRAINER_FACTS, OptimizerBuilder, SchedulerBuilder

if TYPE_CHECKING:
    from src.data.datamodule import DataModule

log = logging.getLogger(__name__)

_OPTIMIZER_CORE_FIELDS = frozenset({"name", "lr", "weight_decay"})
_SCHEDULER_CORE_FIELDS = frozenset({"name", "interval", "frequency", "monitor", "strict", "runtime_kwargs"})


def build_optimizer_builder(
    optimizer_config: OptimizerConfig,
    task_lr_overrides: dict[str, float] | None = None,
) -> OptimizerBuilder:
    """Build an ``OptimizerBuilder`` from the optimizer config.

    Resolves the optimizer class from ``optimizer_config.name`` (so ``sgd`` etc.
    actually take effect) and forwards any extra fields (``momentum``, ``betas``,
    ``nesterov``, ...) as constructor kwargs. Per-task LR overrides (from
    ``build_task_lr_overrides``) are bound here, so the builder is fully configured
    and its consumer just hands it the model.

    Parameters:
        optimizer_config (OptimizerConfig): Validated optimizer config (extras allowed).
        task_lr_overrides (dict[str, float] | None): Per-task LR (task name → lr).

    Returns:
        OptimizerBuilder: Builder bound to the named optimizer class and per-head LRs.
    """
    extra = forward_extras(optimizer_config, _OPTIMIZER_CORE_FIELDS)
    return OptimizerBuilder.from_name(
        name=optimizer_config.name,
        base_lr=optimizer_config.lr,
        base_weight_decay=optimizer_config.weight_decay,
        extra_kwargs=extra,
        task_lr_overrides=task_lr_overrides,
    )


def build_scheduler_builder(scheduler_config: SchedulerConfig | None) -> SchedulerBuilder | None:
    """Build a ``SchedulerBuilder`` from the optional scheduler config.

    ``None`` config → ``None`` builder (constant LR). Validates that every
    ``runtime_kwargs`` value names a known trainer fact (fail-fast, at startup).
    Scheduler-specific extras forward verbatim, exactly like the optimizer.

    Parameters:
        scheduler_config (SchedulerConfig | None): Validated scheduler config (extras allowed).

    Returns:
        SchedulerBuilder | None: Builder bound to the named scheduler, or None.
    """
    if scheduler_config is None:
        return None
    unknown = set(scheduler_config.runtime_kwargs.values()) - set(TRAINER_FACTS)
    if unknown:
        raise ValueError(
            f"scheduler.runtime_kwargs references unknown trainer fact(s) {sorted(unknown)}. "
            f"Available: {sorted(TRAINER_FACTS)}."
        )
    extra = forward_extras(scheduler_config, _SCHEDULER_CORE_FIELDS)
    return SchedulerBuilder.from_name(
        scheduler_config.name,
        interval=scheduler_config.interval,
        frequency=scheduler_config.frequency,
        monitor=scheduler_config.monitor,
        strict=scheduler_config.strict,
        runtime_kwargs=scheduler_config.runtime_kwargs,
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
    return {
        name: task_config.optimizer.lr
        for name, task_config in config.tasks.items()
        if task_config.optimizer is not None
    }


def build_teachers(distillation_config: DistillationConfig, tasks: list[Task]) -> TeacherEnsemble:
    """Assemble the frozen teacher ensemble: teacher backbone + student-derived heads + weights.

    Each teacher's heads are sized from the *student's* task specs, so the teacher's
    per-task logit shapes match the student's by construction. Weights load via the
    shared ``extract_model_state_dict`` (Lightning ``.ckpt`` or raw state dict); the
    LitModule ``model.`` key prefix is stripped when present. EMA checkpoints
    contribute their EMA weights automatically (that is what their ``state_dict`` holds).

    Parameters:
        distillation_config (DistillationConfig): Validated distillation section.
        tasks (list[Task]): Student tasks (their head specs size the teacher heads).

    Returns:
        TeacherEnsemble: Frozen ensemble producing averaged soft targets per task.
    """
    head_specs = {task.name: task.head_spec for task in tasks}
    teachers: list[nn.Module] = []
    for teacher_config in distillation_config.teachers:
        backbone = build_backbone(teacher_config.backbone)
        composite = build_composite_model(backbone, head_specs)
        checkpoint = torch.load(teacher_config.ckpt_path, map_location="cpu", weights_only=True)
        state = extract_model_state_dict(checkpoint)
        # A LitModule checkpoint prefixes model weights with "model."; a raw CompositeModel
        # state dict does not. Strip the prefix in the former case, load verbatim otherwise.
        if any(key.startswith("model.") for key in state):
            state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
        composite.load_state_dict(state, strict=True)
        teachers.append(composite)
    return TeacherEnsemble(teachers)


def resolve_distillation_bricks(
    distillation_config: DistillationConfig, tasks: list[Task]
) -> tuple[dict[str, Criterion], dict[str, float]]:
    """Resolve per-task soft-loss criteria and additive weights; validate task names.

    ``tasks: null`` distills every task. A per-task ``weight`` map applies its values
    (defaulting missing entries to ``1.0``); a scalar ``weight`` applies to all. The
    default criterion is the ``kl_divergence`` brick carrying ``temperature``; an
    explicit ``loss:`` brick-spec wins outright.

    Parameters:
        distillation_config (DistillationConfig): Validated distillation section.
        tasks (list[Task]): Student tasks (the universe of distillable names).

    Returns:
        tuple[dict[str, Criterion], dict[str, float]]: Per-task criteria and weights.

    Raises:
        ValueError: If ``tasks`` or a ``weight`` map names an unknown task.
    """
    task_names = [task.name for task in tasks]
    distilled = distillation_config.tasks if distillation_config.tasks is not None else task_names
    unknown = set(distilled) - set(task_names)
    if unknown:
        raise ValueError(f"distillation.tasks reference unknown task(s): {sorted(unknown)}; tasks: {task_names}.")

    weight = distillation_config.weight
    if isinstance(weight, dict):
        unknown_weights = set(weight) - set(distilled)
        if unknown_weights:
            raise ValueError(f"distillation.weight references unknown task(s): {sorted(unknown_weights)}.")
        weights = {name: weight.get(name, 1.0) for name in distilled}
    else:
        weights = {name: weight for name in distilled}

    def build_criterion() -> Criterion:
        if distillation_config.loss is None:
            return criteria.create("kl_divergence", temperature=distillation_config.temperature)
        return instantiate(distillation_config.loss, criteria)

    return {name: build_criterion() for name in distilled}, weights


def build_lit_module(
    config: ExperimentConfig,
    model: CompositeModel,
    tasks: list[Task],
    optimizer_builder: OptimizerBuilder,
    scheduler_builder: SchedulerBuilder | None = None,
) -> BaseLitModule:
    """Build the training module from the assembled collaborators and config hyperparams.

    Per-task LR overrides already live on ``optimizer_builder`` (bound in
    ``build_optimizer_builder``), so this only wires the module and serialises the
    full config as hyperparams the logger records at ``on_fit_start``. When
    ``config.distillation`` is set, the distillation regime is built instead of the
    plain module (the sanctioned single branch, like ``kind: multi`` for backbones).

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        model (CompositeModel): Backbone + heads.
        tasks (list[Task]): Assembled task bundles.
        optimizer_builder (OptimizerBuilder): Fully configured (incl. per-head LRs).

    Returns:
        BaseLitModule: ``DistillationLitModule`` when distillation is configured, else ``LitModule``.
    """
    hparams = config.model_dump(mode="json")
    if config.distillation is not None:
        distillation_criteria, distillation_weights = resolve_distillation_bricks(config.distillation, tasks)
        return DistillationLitModule(
            model=model,
            tasks=tasks,
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            teachers=build_teachers(config.distillation, tasks),
            distillation_criteria=distillation_criteria,
            distillation_weights=distillation_weights,
            hparams=hparams,
        )
    return LitModule(
        model=model,
        tasks=tasks,
        optimizer_builder=optimizer_builder,
        scheduler_builder=scheduler_builder,
        hparams=hparams,
    )


def build_lit_data_module(data_module: DataModule) -> LitDataModule:
    """Wrap the domain ``DataModule`` in its Lightning humble object.

    The data-side counterpart of ``build_lit_module``: the plain ``DataModule``
    owns all data logic (sources, encoders, cache, dataloader knobs); ``LitDataModule``
    is the thin Lightning adapter exposing train/val/test dataloaders. Build it after
    ``data_module.setup()`` so encoders are fitted and ``num_classes`` is populated.

    Parameters:
        data_module (DataModule): The setup-complete domain data module.

    Returns:
        LitDataModule: Lightning-facing wrapper ready for ``Trainer.fit``.
    """
    return LitDataModule(data_module)


def build_logger(config: ExperimentConfig) -> Any:
    """Build the experiment logger by dispatching on ``logger.kind`` via ``logger_builders``.

    Returns ``False`` (Lightning's "disable logging" sentinel) for ``kind: none``;
    a concrete ``Logger`` for any registered backend. The single home for the dispatch
    (the registry holds only the per-backend builders).

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        Logger | bool: Configured logger, or ``False`` to disable.

    Raises:
        ValueError: If ``config.logger.kind`` is not a registered backend.
    """
    from src.loggers.registry import logger_builders

    kind = config.logger.kind
    if kind not in logger_builders:
        known = ", ".join(sorted(str(key) for key in logger_builders.keys()))
        raise ValueError(f"Unknown logger kind: {kind!r}. Known kinds: {known}.")
    return logger_builders.create(kind, config)


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


def _lightning_prints_test_results(trainer: L.Trainer) -> bool:
    """Whether ``trainer.test`` should print Lightning's own results table.

    Suppressed when our ``MetricsProgressBar`` is active: it already renders the test
    metrics (vector metrics as their mean) in the live table, so Lightning's verbose
    per-class dump would be redundant clutter. Without our bar, keep the default table.
    """
    # ``Trainer.callbacks`` is a public runtime attribute the type stubs do not expose.
    callbacks: list[Any] = getattr(trainer, "callbacks", [])
    return not any(isinstance(callback, MetricsProgressBar) for callback in callbacks)


def run_experiment(
    trainer: L.Trainer,
    lit_module: BaseLitModule,
    lit_data_module: LitDataModule,
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
        lit_module (BaseLitModule): Model module (any regime: standard or distillation).
        lit_data_module (LitDataModule): Data module.
        config (ExperimentConfig): Validated experiment config.
        tasks (list[Task]): Active tasks (for export planning).
    """
    trained = False
    tested = False

    if config.run_train:
        if config.init_ckpt_path is not None:
            load_init_weights(lit_module, config.init_ckpt_path)
        trainer.fit(lit_module, lit_data_module)
        trained = True
        log.info("Training complete.")

    if config.run_test:
        ckpt_path = resolve_test_ckpt_path(trainer, config, trained=trained)
        verbose = _lightning_prints_test_results(trainer)
        if ckpt_path is not None:
            log.info("Running test with checkpoint: %s", ckpt_path)
            trainer.test(lit_module, lit_data_module, ckpt_path=ckpt_path, verbose=verbose)
        else:
            log.info("Running test with in-memory weights.")
            trainer.test(lit_module, lit_data_module, verbose=verbose)
        tested = True
        log.info("Test complete.")

    run_export(trainer, lit_module, tasks, config, trained=trained, tested=tested)
