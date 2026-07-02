"""Data-layer wiring: transforms, data sources, and the DataModule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.composition.wiring.common import forward_extras
from src.config.schema import CacheConfig, DataConfig, ExperimentConfig, SourceConfig
from src.core.enums import Stage
from src.core.instantiate import instantiate
from src.core.runtime import RuntimeContext
from src.data.bindings import SourceBinding, TargetBinding
from src.data.cache import BYTES_PER_GIB
from src.data.datamodule import CacheOptions, DataLoaderOptions, DataModule
from src.data.registry import data_sources
from src.data.sources import DataSource
from src.transforms.sample import AlbumentationsTransform, Transform

log = logging.getLogger(__name__)

# File extension → data_sources registry key, for inferring the source format.
_EXTENSION_TO_SOURCE: dict[str, str] = {".csv": "csv", ".json": "json"}

# Alias of the schema default (single home) — used only when caching is disabled, so workers is inert.
_DEFAULT_CACHE_WORKERS: int = CacheConfig.model_fields["workers"].default

# DataLoader knobs the builder passes explicitly; every other (extra) key is forwarded
# verbatim to torch.utils.data.DataLoader (the reserved keys are rejected in the schema).
_DATALOADER_CORE_FIELDS = frozenset({"num_workers", "pin_memory", "persistent_workers", "drop_last", "prefetch_factor"})


def _resolve_cache_bytes(cache: CacheConfig | None) -> int | None:
    """Compute the cache byte budget = min(ram_fraction · available RAM, max_gb), and log it.

    Logs which term bound the budget — so it's obvious when the effective budget
    is the RAM fraction rather than the ``max_gb`` ceiling (a common surprise).
    """
    if cache is None or cache.ram_fraction <= 0:
        return None
    import psutil

    available = psutil.virtual_memory().available
    fraction_bytes = int(cache.ram_fraction * available)
    cap_bytes = int(cache.max_gb * BYTES_PER_GIB) if cache.max_gb is not None else fraction_bytes
    budget = min(fraction_bytes, cap_bytes)
    limiter = "max_gb cap" if cap_bytes < fraction_bytes else f"{cache.ram_fraction:.0%} of available RAM"
    log.info(
        "Data cache budget: %.2f GiB (limited by %s; available RAM %.2f GiB, max_gb %s).",
        budget / BYTES_PER_GIB,
        limiter,
        available / BYTES_PER_GIB,
        f"{cache.max_gb:.2f} GiB" if cache.max_gb is not None else "none",
    )
    return budget


def build_transforms(
    config: ExperimentConfig,
    bindings: list[TargetBinding] | None = None,
) -> dict[Stage, Transform]:
    """Build per-stage input transforms from the experiment config.

    When ``config.transforms`` contains per-stage Albumentations pipeline specs,
    each stage is instantiated via ``instantiate`` and wrapped in an
    ``AlbumentationsTransform``. Missing stages fall back to the nearest eval
    transform (val → test → predict).

    ``bindings`` is used to register spatial target keys (masks) via
    ``add_targets`` so every geometric op is applied to image and mask together.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        bindings (list[TargetBinding] | None): Target bindings; ``None`` → no masks.

    Returns:
        dict[Stage, Transform]: A transform for every lifecycle stage.
    """
    spatial = [binding.name for binding in (bindings or []) if binding.encoder.spatial]

    if config.transforms is None:
        raise ValueError(
            "transforms config is required. Add a transforms group to your experiment config, "
            "e.g. 'defaults: [transforms: default]' or set 'transforms:' inline."
        )
    return _build_transforms_from_config(config.transforms, spatial)


def _build_stage_transform(spec: Any, spatial_targets: list[str]) -> Transform:
    """Instantiate one stage's transform spec.

    A spec may instantiate directly to a ``Transform`` (e.g. ``IdentityTransform`` for the
    embedding modality) — used as-is — or to an Albumentations ``Compose``, wrapped in an
    ``AlbumentationsTransform`` (registering the spatial mask targets).
    """
    built = instantiate(spec)
    if isinstance(built, Transform):
        return built
    return AlbumentationsTransform(built, spatial_targets=spatial_targets)


def _build_transforms_from_config(
    transforms_config: dict[str, Any],
    spatial_targets: list[str],
) -> dict[Stage, Transform]:
    """Instantiate one ``Transform`` per stage from its spec, deriving missing eval stages."""
    result: dict[Stage, Transform] = {
        Stage(stage_str): _build_stage_transform(spec, spatial_targets) for stage_str, spec in transforms_config.items()
    }
    # Derive missing stages from the nearest eval transform.
    eval_transform = result.get(Stage.VAL) or result.get(Stage.TEST)
    if eval_transform:
        for stage in (Stage.TEST, Stage.PREDICT):
            result.setdefault(stage, eval_transform)
    return result


def build_staged_sources(
    config: ExperimentConfig,
    global_transforms: dict[Stage, Transform],
    target_bindings: list[TargetBinding],
) -> dict[Stage, list[SourceBinding]] | None:
    """Build per-stage source bindings (pre-split mode); ``None`` in split mode.

    Each stage maps to a list of ``SourceBinding`` — one per source declared under that stage,
    carrying that source's transform *for that stage* (its single override if a ``SourceConfig``
    declares one, else the global stage transform). The ``DataModule`` reads each source and
    combines a stage's per-source ``Dataset``s with ``ConcatDataset``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        global_transforms (dict[Stage, Transform]): Global per-stage transforms (fallback).
        target_bindings (list[TargetBinding]): Used to register spatial mask targets.

    Returns:
        dict[Stage, list[SourceBinding]] | None: Per-stage source bindings, or ``None`` in split mode.
    """
    data_config = config.data
    if not isinstance(data_config.sources, dict):
        return None
    spatial_targets = [binding.name for binding in target_bindings if binding.encoder.spatial]
    result: dict[Stage, list[SourceBinding]] = {}
    for stage_str, value in data_config.sources.items():
        stage = Stage(stage_str)
        bindings: list[SourceBinding] = []
        for entry in _stage_entries(value):
            source = _create_source(_entry_paths(entry), data_config.source_type)
            transform = _resolve_staged_transform(entry, stage, global_transforms, spatial_targets)
            bindings.append(SourceBinding(source=source, transforms={stage: transform}))
        result[stage] = bindings
    return result


def _stage_entries(value: str | SourceConfig | list[str | SourceConfig]) -> list[str | SourceConfig]:
    """Normalise one pre-split stage value (path / ``SourceConfig`` / list) to a list of entries."""
    if isinstance(value, (str, SourceConfig)):
        return [value]
    return list(value)


def _resolve_staged_transform(
    entry: str | SourceConfig,
    stage: Stage,
    global_transforms: dict[Stage, Transform],
    spatial_targets: list[str],
) -> Transform:
    """One pre-split source's transform for its stage: its single override, else the global one."""
    if isinstance(entry, SourceConfig) and entry.transforms is not None:
        return _build_stage_transform(entry.transforms, spatial_targets)
    return global_transforms[stage]


def build_data_source(data_config: DataConfig) -> DataSource:
    """Build a single ``DataSource`` when ``sources`` is a path string or list (split mode).

    The registry key comes from ``source_type`` when set, otherwise inferred from
    the file extension.

    Parameters:
        data_config (DataConfig): Validated data config (split mode).

    Returns:
        DataSource: A source ready for ``DataModule.setup`` to ``read``.

    Raises:
        ValueError: If the format cannot be inferred or paths mix extensions.
    """
    assert not isinstance(data_config.sources, dict), "Use build_staged_sources for pre-split mode."
    entries = [data_config.sources] if isinstance(data_config.sources, str) else list(data_config.sources)
    paths = [path for entry in entries for path in _entry_paths(entry)]
    return _create_source(paths, data_config.source_type)


def _entry_paths(entry: str | SourceConfig) -> list[str]:
    """The annotation path(s) of one ``sources`` list entry (string path or ``SourceConfig``)."""
    path = entry.path if isinstance(entry, SourceConfig) else entry
    return [path] if isinstance(path, str) else list(path)


def _create_source(paths: list[str], source_type: str | None) -> DataSource:
    """Build a ``DataSource`` over ``paths`` (registry key from ``source_type`` or inferred)."""
    key = source_type or _infer_source_type(paths)
    return data_sources.create(key, paths)


def _resolve_source_transforms(
    overrides: dict[str, Any] | None,
    global_transforms: dict[Stage, Transform],
    spatial_targets: list[str],
) -> dict[Stage, Transform]:
    """The global stage transforms with the source's per-stage overrides replacing them.

    Replace semantics: only the stages the source declares are overridden; every unset stage
    falls back to the global transform. No eval-stage derivation here — that would override
    stages the source did not ask for (which should stay global).
    """
    resolved = dict(global_transforms)
    for stage_str, spec in (overrides or {}).items():
        resolved[Stage(stage_str)] = _build_stage_transform(spec, spatial_targets)
    return resolved


def build_source_bindings(
    config: ExperimentConfig,
    global_transforms: dict[Stage, Transform],
    target_bindings: list[TargetBinding],
) -> list[SourceBinding]:
    """Build per-source bindings for split mode: each source plus its resolved per-stage transforms.

    With no per-source override the whole list collapses to a **single** binding over one
    combined source — the current behaviour (one frame, global split). When any source declares
    its own ``transforms``, each source becomes its **own** binding (read + split independently,
    combined per stage via ``ConcatDataset``) with override-or-global transforms per stage.

    Parameters:
        config (ExperimentConfig): Validated experiment config (split mode).
        global_transforms (dict[Stage, Transform]): The global per-stage transforms (fallback).
        target_bindings (list[TargetBinding]): Used to register spatial mask targets on per-source
            Albumentations pipelines.

    Returns:
        list[SourceBinding]: One binding (no overrides) or one per source (with overrides).
    """
    data_config = config.data
    assert not isinstance(data_config.sources, dict), "Use build_staged_sources for pre-split mode."
    entries: list[str | SourceConfig] = (
        [data_config.sources] if isinstance(data_config.sources, str) else list(data_config.sources)
    )
    spatial_targets = [binding.name for binding in target_bindings if binding.encoder.spatial]

    if not any(isinstance(entry, SourceConfig) and entry.transforms for entry in entries):
        paths = [path for entry in entries for path in _entry_paths(entry)]
        source = _create_source(paths, data_config.source_type)
        return [SourceBinding(source=source, transforms=dict(global_transforms))]

    bindings: list[SourceBinding] = []
    for entry in entries:
        overrides = entry.transforms if isinstance(entry, SourceConfig) else None
        source = _create_source(_entry_paths(entry), data_config.source_type)
        transforms = _resolve_source_transforms(overrides, global_transforms, spatial_targets)
        bindings.append(SourceBinding(source=source, transforms=transforms))
    return bindings


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
    that encoder fitting and dataset construction happen in the right order
    relative to ``build_tasks``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        bindings (list[TargetBinding]): Target bindings (un-fitted encoders).
        runtime (RuntimeContext): Populated by ``DataModule.setup()``.

    Returns:
        DataModule: Ready to call ``.setup()`` on.
    """
    global_transforms = build_transforms(config, bindings)
    staged = build_staged_sources(config, global_transforms, bindings)
    source_bindings = build_source_bindings(config, global_transforms, bindings) if staged is None else None
    dataloader_config = config.dataloader
    return DataModule(
        target_bindings=bindings,
        inputs_config=config.data.inputs,
        transforms=global_transforms,
        runtime=runtime,
        batch_size=config.batch_size,
        seed=config.seed,
        source_bindings=source_bindings,
        split=config.data.split if staged is None else None,
        split_stratify=config.data.split_stratify if staged is None else None,
        max_samples=config.data.max_samples,
        staged_sources=staged,
        dataloader_options=DataLoaderOptions(
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=dataloader_config.persistent_workers,
            drop_last=dataloader_config.drop_last,
            prefetch_factor=dataloader_config.prefetch_factor,
            extra_kwargs=forward_extras(dataloader_config, _DATALOADER_CORE_FIELDS),
        ),
        root_path=config.data.root_path,
        cache_options=CacheOptions(
            max_bytes=_resolve_cache_bytes(config.data.cache),
            workers=config.data.cache.workers if config.data.cache is not None else _DEFAULT_CACHE_WORKERS,
        ),
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
