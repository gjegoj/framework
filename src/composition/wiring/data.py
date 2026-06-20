"""Data-layer wiring: transforms, data sources, and the DataModule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.composition.wiring.common import forward_extras
from src.config.schema import CacheConfig, DataConfig, ExperimentConfig
from src.core.enums import Stage
from src.core.instantiate import instantiate
from src.core.runtime import RuntimeContext
from src.data.bindings import TargetBinding
from src.data.cache import BYTES_PER_GIB
from src.data.datamodule import CacheOptions, DataLoaderOptions, DataModule
from src.data.registry import data_sources
from src.data.sources import DataSource
from src.transforms.sample import AlbumentationsTransform, Transform

log = logging.getLogger(__name__)

# File extension → data_sources registry key, for inferring the source format.
_EXTENSION_TO_SOURCE: dict[str, str] = {".csv": "csv", ".json": "json"}

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


def _build_transforms_from_config(
    transforms_config: dict[str, Any],
    spatial_targets: list[str],
) -> dict[Stage, Transform]:
    """Instantiate one ``Transform`` per stage from its spec.

    A spec may instantiate directly to a ``Transform`` (e.g. ``IdentityTransform``
    for the embedding modality) — used as-is — or to an Albumentations ``Compose``,
    which is wrapped in an ``AlbumentationsTransform``.
    """
    result: dict[Stage, Transform] = {}
    for stage_str, spec in transforms_config.items():
        built = instantiate(spec)
        if not isinstance(built, Transform):
            built = AlbumentationsTransform(built, spatial_targets=spatial_targets)
        result[Stage(stage_str)] = built

    # Derive missing stages from the nearest eval transform.
    eval_transform = result.get(Stage.VAL) or result.get(Stage.TEST)
    if eval_transform:
        for stage in (Stage.TEST, Stage.PREDICT):
            if stage not in result:
                result[stage] = eval_transform
    return result


def build_staged_sources(data_config: DataConfig) -> dict[Stage, DataSource] | None:
    """Build per-stage ``DataSource`` objects when ``sources`` is a dict (pre-split mode).

    Returns ``None`` in split mode (``sources`` is a str/list), so the caller
    can branch cleanly::

        staged = build_staged_sources(config.data)
        data_module = DataModule(
            source=build_data_source(config.data) if staged is None else None,
            split=config.data.split if staged is None else None,
            staged_sources=staged,
            ...
        )

    Parameters:
        data_config (DataConfig): Validated data config.

    Returns:
        dict[Stage, DataSource] or None: Per-stage sources, or ``None`` in split mode.
    """
    if not isinstance(data_config.sources, dict):
        return None
    result: dict[Stage, DataSource] = {}
    for stage_str, paths in data_config.sources.items():
        path_list = [paths] if isinstance(paths, str) else list(paths)
        key = data_config.source_type or _infer_source_type(path_list)
        result[Stage(stage_str)] = data_sources.create(key, path_list)
    return result


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
    paths = [data_config.sources] if isinstance(data_config.sources, str) else list(data_config.sources)
    key = data_config.source_type or _infer_source_type(paths)
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
    that encoder fitting and dataset construction happen in the right order
    relative to ``build_tasks``.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        bindings (list[TargetBinding]): Target bindings (un-fitted encoders).
        runtime (RuntimeContext): Populated by ``DataModule.setup()``.

    Returns:
        DataModule: Ready to call ``.setup()`` on.
    """
    staged = build_staged_sources(config.data)
    dataloader_config = config.dataloader
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
            workers=config.data.cache.workers if config.data.cache is not None else 8,
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
