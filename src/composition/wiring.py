"""Wiring helpers: pure functions that map a validated config to runtime objects.

Extracted from main.py so they can be unit-tested without Hydra. The composition
order is enforced by the function signatures: ``build_tasks`` requires a
``RuntimeContext`` already populated by ``DataModule.setup()``.
"""

from __future__ import annotations

from src.config.schema import ExperimentConfig, OptimizerConfig, TaskConfig
from src.core.entities import Task
from src.core.enums import Stage
from src.core.instantiate import instantiate
from src.core.runtime import RuntimeContext
from src.data.bindings import TargetBinding
from src.data.codecs import TargetCodec, target_codecs
from src.data.transforms import Transform, build_basic_transform
from src.tasks.presets import task_presets
from src.tasks.strategies.objective import objective_strategies
from src.training.optimizer import OptimizerBuilder


def build_transforms(config: ExperimentConfig) -> dict[Stage, Transform]:
    """Build per-stage input transforms from the experiment config.

    Train and eval use the same pipeline until albumentations is wired;
    augmentation is added later via the albumentations path.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        dict[Stage, Transform]: A transform for every lifecycle stage.
    """
    train_t = build_basic_transform(config.image_size, config.mean, config.std)
    eval_t = build_basic_transform(config.image_size, config.mean, config.std)
    return {
        Stage.TRAIN: train_t,
        Stage.VAL: eval_t,
        Stage.TEST: eval_t,
        Stage.PREDICT: eval_t,
    }


def _resolve_codec(task_cfg: TaskConfig) -> TargetCodec:
    """Build the data-layer target codec for one task.

    Priority: explicit ``target_codec:`` spec > the objective's ``default_codec``.
    The objective is the authority on label encoding, so the default follows from
    the resolved objective (preset default unless overridden in config).

    Parameters:
        task_cfg (TaskConfig): Validated task config.

    Returns:
        TargetCodec: An un-fitted codec ready for ``DataModule.setup``.
    """
    if task_cfg.target_codec is not None:
        return instantiate(task_cfg.target_codec, target_codecs)

    objective = task_presets.create(task_cfg.preset).resolve_objective(task_cfg.objective)
    codec_key = objective_strategies.create(objective).default_codec
    return instantiate(codec_key, target_codecs)


def build_bindings(config: ExperimentConfig) -> list[TargetBinding]:
    """Build target bindings (task name → column → codec) for all tasks.

    Called before ``DataModule.setup()`` — codecs are un-fitted here and fitted
    inside ``setup()``. The data-codec follows from the task's objective unless
    overridden by ``target_codec:`` in the task config.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        list[TargetBinding]: One binding per task, in declaration order.
    """
    return [
        TargetBinding(name=task_name, column=task_cfg.target, codec=_resolve_codec(task_cfg))
        for task_name, task_cfg in config.tasks.items()
    ]


def _resolve_num_classes(task_name: str, task_cfg: TaskConfig, runtime: RuntimeContext) -> int:
    """Return the concrete class count / output dim for a task.

    For regression tasks with ``dim`` set, returns ``dim`` directly.
    For all others tries ``num_classes`` from config then ``RuntimeContext``.
    """
    if task_cfg.dim is not None:
        return task_cfg.dim
    value = task_cfg.num_classes or runtime.num_classes.get(task_name)
    if value is None:
        raise ValueError(
            f"num_classes for task '{task_name}' is not set in config and could not be "
            "inferred from data. Ensure DataModule.setup() ran before build_tasks(), "
            "or set num_classes / dim explicitly in the task config."
        )
    return value


def build_tasks(config: ExperimentConfig, runtime: RuntimeContext) -> list[Task]:
    """Build task bundles after ``DataModule.setup()`` has populated ``RuntimeContext.num_classes``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        runtime (RuntimeContext): Populated context (num_classes must be set for each task).

    Returns:
        list[Task]: Assembled task bundles in config declaration order.

    Raises:
        ValueError: If num_classes for any task cannot be resolved.
    """
    tasks: list[Task] = []
    for task_name, task_cfg in config.tasks.items():
        num_classes = _resolve_num_classes(task_name, task_cfg, runtime)
        preset = task_presets.create(task_cfg.preset)
        task = preset.build(
            name=task_name,
            num_classes=num_classes,
            objective=task_cfg.objective,
            weight=task_cfg.weight,
            loss=task_cfg.loss,
            metrics=task_cfg.metrics,
        )
        tasks.append(task)
    return tasks


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
    extra = {key: value for key, value in optimizer_cfg.model_dump().items() if key not in _OPTIMIZER_CORE_FIELDS}
    return OptimizerBuilder.from_name(
        name=optimizer_cfg.name,
        base_lr=optimizer_cfg.lr,
        base_weight_decay=optimizer_cfg.weight_decay,
        extra_kwargs=extra,
    )
