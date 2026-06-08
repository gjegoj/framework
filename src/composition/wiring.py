"""Wiring helpers: pure functions that map a validated config to runtime objects.

Extracted from main.py so they can be unit-tested without Hydra. The composition
order is enforced by the function signatures: ``build_tasks`` requires a
``RuntimeContext`` already populated by ``DataModule.setup()``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from src.config.schema import BackboneConfig, DataConfig, ExperimentConfig, OptimizerConfig, TaskConfig
from src.core.entities import Task
from src.core.enums import Stage
from src.core.instantiate import instantiate
from src.core.ports import Backbone
from src.core.runtime import RuntimeContext
from src.data.bindings import TargetBinding
from src.data.codecs import TargetCodec, target_codecs
from src.data.datamodule import DataModule
from src.data.sources import DataSource, data_sources
from src.data.transforms import AlbumentationsTransform, Transform
from src.models.assembly import CompositeModel
from src.models.registry import backbones
from src.tasks.presets import task_presets
from src.tasks.strategies.objective import objective_strategies
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder

_BACKBONE_CORE_FIELDS = frozenset({"kind", "name", "pretrained"})

# File extension → data_sources registry key, for inferring the source format.
_EXTENSION_TO_SOURCE: dict[str, str] = {".csv": "csv", ".json": "json"}


def build_transforms(
    config: ExperimentConfig,
    bindings: list[TargetBinding] | None = None,
) -> dict[Stage, Transform]:
    """Build per-stage input transforms from the experiment config.

    When ``config.transforms`` contains per-stage Albumentations pipeline specs,
    each stage is instantiated via ``instantiate`` and wrapped in an
    ``AlbumentationsTransform``. Missing stages fall back to the nearest eval
    transform (val → test → predict).

    When ``config.transforms`` is ``None``, the default resize + normalize +
    ToTensorV2 pipeline is built from ``image_size`` / ``mean`` / ``std``.

    ``bindings`` is used to register spatial target keys (masks) via
    ``add_targets`` so every geometric op is applied to image and mask together.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        bindings (list[TargetBinding] | None): Target bindings; ``None`` → no masks.

    Returns:
        dict[Stage, Transform]: A transform for every lifecycle stage.
    """
    spatial = [b.name for b in (bindings or []) if b.codec.spatial]

    if config.transforms is None:
        raise ValueError(
            "transforms config is required. Add a transforms group to your experiment config, "
            "e.g. 'defaults: [transforms: default]' or set 'transforms:' inline."
        )
    return _build_transforms_from_config(config.transforms, spatial)


def _build_transforms_from_config(
    transforms_cfg: dict[str, Any],
    spatial_targets: list[str],
) -> dict[Stage, Transform]:
    """Instantiate per-stage ``AlbumentationsTransform`` from ``_target_`` specs."""
    result: dict[Stage, Transform] = {}
    for stage_str, spec in transforms_cfg.items():
        compose = instantiate(spec)
        result[Stage(stage_str)] = AlbumentationsTransform(compose, spatial_targets=spatial_targets)

    # Derive missing stages from the nearest eval transform.
    eval_t = result.get(Stage.VAL) or result.get(Stage.TEST)
    if eval_t:
        for stage in (Stage.TEST, Stage.PREDICT):
            if stage not in result:
                result[stage] = eval_t
    return result


def build_backbone(backbone_cfg: BackboneConfig) -> Backbone:
    """Build the backbone from config, forwarding adapter-specific extras.

    ``kind`` selects the registry adapter; ``name``/``pretrained`` are passed
    explicitly and any extra fields (e.g. smp's ``encoder_name``) are forwarded
    as keyword args.

    Parameters:
        backbone_cfg (BackboneConfig): Validated backbone config (extras allowed).

    Returns:
        Backbone: The constructed backbone adapter.
    """
    extra = {key: value for key, value in backbone_cfg.model_dump().items() if key not in _BACKBONE_CORE_FIELDS}
    return backbones.create(backbone_cfg.kind, name=backbone_cfg.name, pretrained=backbone_cfg.pretrained, **extra)


def build_staged_sources(data_cfg: DataConfig) -> dict[Stage, DataSource] | None:
    """Build per-stage ``DataSource`` objects when ``sources`` is a dict (pre-split mode).

    Returns ``None`` in split mode (``sources`` is a str/list), so the caller
    can branch cleanly::

        staged = build_staged_sources(config.data)
        dm = DataModule(
            source=build_data_source(config.data) if staged is None else None,
            split=config.data.split if staged is None else None,
            staged_sources=staged,
            ...
        )

    Parameters:
        data_cfg (DataConfig): Validated data config.

    Returns:
        dict[Stage, DataSource] or None: Per-stage sources, or ``None`` in split mode.
    """
    if not isinstance(data_cfg.sources, dict):
        return None
    result: dict[Stage, DataSource] = {}
    for stage_str, paths in data_cfg.sources.items():
        path_list = [paths] if isinstance(paths, str) else list(paths)
        key = data_cfg.source_type or _infer_source_type(path_list)
        result[Stage(stage_str)] = data_sources.create(key, path_list)
    return result


def build_data_source(data_cfg: DataConfig) -> DataSource:
    """Build a single ``DataSource`` when ``sources`` is a path string or list (split mode).

    The registry key comes from ``source_type`` when set, otherwise inferred from
    the file extension.

    Parameters:
        data_cfg (DataConfig): Validated data config (split mode).

    Returns:
        DataSource: A source ready for ``DataModule.setup`` to ``read``.

    Raises:
        ValueError: If the format cannot be inferred or paths mix extensions.
    """
    assert not isinstance(data_cfg.sources, dict), "Use build_staged_sources for pre-split mode."
    paths = [data_cfg.sources] if isinstance(data_cfg.sources, str) else list(data_cfg.sources)
    key = data_cfg.source_type or _infer_source_type(paths)
    return data_sources.create(key, paths)


def build_data_module(
    config: ExperimentConfig,
    bindings: list[TargetBinding],
    runtime: RuntimeContext,
) -> DataModule:
    """Build and return a configured ``DataModule`` from the experiment config.

    Handles both data modes transparently: when ``config.data.sources`` is a
    dict the module uses pre-split sources; otherwise it reads one source and
    splits by ratio (with optional stratification).

    ``DataModule.setup()`` must be called by the caller after this returns so
    that codec fitting and dataset construction happen in the right order
    relative to ``build_tasks``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        bindings (list[TargetBinding]): Target bindings (un-fitted codecs).
        runtime (RuntimeContext): Populated by ``DataModule.setup()``.

    Returns:
        DataModule: Ready to call ``.setup()`` on.
    """
    staged = build_staged_sources(config.data)
    dl = config.dataloader
    return DataModule(
        target_bindings=bindings,
        inputs_config=config.data.inputs,
        transforms=build_transforms(config, bindings),
        runtime=runtime,
        batch_size=config.batch_size,
        seed=config.seed,
        source=build_data_source(config.data) if staged is None else None,
        split=config.data.split if staged is None else None,
        split_stratify=config.data.split_stratify if staged is None else None,
        max_samples=config.data.max_samples,
        staged_sources=staged,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
        persistent_workers=dl.persistent_workers,
        drop_last=dl.drop_last,
        prefetch_factor=dl.prefetch_factor,
        root_path=config.data.root_path,
    )


def _infer_source_type(paths: list[str]) -> str:
    """Infer the ``data_sources`` key from a consistent file extension."""
    extensions = {Path(path).suffix.lower() for path in paths}
    if len(extensions) != 1:
        raise ValueError(f"Cannot infer source_type from mixed extensions {sorted(extensions)}; set data.source_type.")
    extension = extensions.pop()
    try:
        return _EXTENSION_TO_SOURCE[extension]
    except KeyError as error:
        known = sorted(_EXTENSION_TO_SOURCE)
        raise ValueError(f"Unknown source extension {extension!r}. Known: {known}; or set data.source_type.") from error


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

    preset = task_presets.create(task_cfg.preset)
    if preset.default_codec is not None:
        codec_key = preset.default_codec
    else:
        objective = preset.resolve_objective(task_cfg.objective)
        codec_key = objective_strategies.create(objective).default_codec

    injected: dict[str, Any] = {}
    if task_cfg.class_mapping is not None:
        injected["class_mapping"] = task_cfg.class_mapping
    return instantiate(codec_key, target_codecs, **injected)


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
            head=task_cfg.head,
            feature_key=task_cfg.feature_key,
        )
        if task_cfg.class_mapping is not None:
            class_names = [task_cfg.class_mapping[i] for i in sorted(task_cfg.class_mapping)]
            task = dataclasses.replace(task, class_names=class_names)
        tasks.append(task)
    return tasks


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


def build_callbacks(config: ExperimentConfig) -> list[Any]:
    """Build the ordered callback list from config.

    Each key in ``config.callbacks`` is looked up in ``callback_registry``
    (or resolved via ``_target_``); its value dict is forwarded as kwargs.
    YAML declaration order controls callback registration order — put ``ema``
    before ``checkpoint`` so EMA weights are active when the checkpoint fires.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        list: Lightning callbacks, ready for ``Trainer(callbacks=...)``.
    """
    if config.callbacks is None:
        return []

    import importlib

    from src.callbacks.registry import callback_registry

    callbacks: list[Any] = []
    for name, raw_params in config.callbacks.items():
        params = dict(raw_params or {})
        target = params.pop("_target_", None)
        if target is not None:
            mod_path, attr = str(target).rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_path), attr)
            callbacks.append(cls(**params))
            continue
        registry_key = str(params.pop("name", name))
        callbacks.append(callback_registry.create(registry_key, **params))
    return callbacks


_OPTIMIZER_CORE_FIELDS = frozenset({"name", "lr", "weight_decay"})


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
