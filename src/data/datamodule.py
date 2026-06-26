"""DataModule: framework-agnostic orchestration of the data pipeline.

Deliberately not a ``LightningDataModule`` — it is plain and unit-testable. The
thin Lightning wrapper that delegates here is added in the training layer.

On ``setup`` it: reads the source, fits target encoders (inferring ``num_classes``
into the RuntimeContext), resolves ``InputBinding``s with loader auto-detection,
splits into per-stage datasets, and records dataset sizes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.bindings import InputBinding, SourceBinding, TargetBinding
from src.data.cache import BYTES_PER_GIB, ArrayCache, CachingLoader, caching_target_encoder
from src.data.collate import collate_samples
from src.data.dataset import Dataset, resolve_path
from src.data.loaders import infer_loader_key, normalize_inputs
from src.data.registry import input_loaders
from src.data.sources import DataSource
from src.data.split import split_dataframe
from src.data.statistics import DatasetStatistics, Distribution, SupportsSummary
from src.transforms.sample import Transform

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataLoaderOptions:
    """DataLoader construction knobs shared across all stages.

    Groups the six DataLoader parameters so they travel as one cohesive value
    instead of widening every ``DataModule`` signature; built from the
    ``DataLoaderConfig`` section in the composition root.

    Parameters:
        num_workers (int): Worker processes per DataLoader (0 → main process).
        pin_memory (bool): Pin host memory for faster CPU→GPU transfers.
        persistent_workers (bool): Keep workers alive between epochs (needs num_workers > 0).
        drop_last (bool): Drop the last incomplete batch during training.
        prefetch_factor (int | None): Batches prefetched per worker (None → PyTorch default).
        extra_kwargs (Mapping[str, Any]): Extra kwargs forwarded verbatim to every
            ``DataLoader`` (e.g. ``timeout``); framework-owned keys always win over these.
    """

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    prefetch_factor: int | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CacheOptions:
    """In-RAM image/mask cache budget for ``DataModule``.

    Parameters:
        max_bytes (int | None): Total RAM budget in bytes; ``None`` or ``<= 0`` disables the cache.
        workers (int): Threads used to warm the cache in the parent process.
    """

    max_bytes: int | None = None
    workers: int = 8


class DataModule:
    """Builds datasets and dataloaders, and populates the RuntimeContext.

    Two construction modes (mutually exclusive):

    **Split mode** — one or more sources, DataModule splits each by ratio::

        DataModule(source=csv_source, split={Stage.TRAIN: 0.8, ...}, ...)        # single source
        DataModule(source_bindings=[SourceBinding(src, transforms), ...], split={...}, ...)  # per-source

    Each source is read and split independently by the same ratios; a stage is the
    ``ConcatDataset`` of one ``Dataset`` per source, each carrying its own transform. The
    ``source`` argument is the single-source shorthand for one ``SourceBinding`` over the
    global ``transforms``.

    **Pre-split mode** — caller provides one ``DataSource`` per stage::

        DataModule(staged_sources={Stage.TRAIN: train_src, Stage.VAL: val_src}, ...)

    In pre-split mode encoders are fitted on train data only (no leakage); in split mode
    they are fitted on the combined train data across all sources.
    ``InputBinding`` loaders are auto-detected from column values at ``setup()``
    time when not explicitly specified in ``inputs_config``.

    Parameters:
        target_bindings (list[TargetBinding]): Per-task target column + encoder.
        inputs_config: Input column(s) and optional loader keys.
            ``str`` → single image input; ``dict`` → multiple inputs.
        transforms (dict[Stage, Transform]): Global per-stage transforms (used by pre-split
            mode and the single-source ``source`` shorthand).
        runtime (RuntimeContext): Populated with num_classes and dataset sizes.
        batch_size (int): Batch size for all stages.
        seed (int): Split seed (split mode only).
        source (DataSource | None): Single-source split-mode shorthand (→ one ``SourceBinding``).
        source_bindings (list[SourceBinding] | None): Per-source split-mode sources, each with
            its own resolved per-stage transforms.
        split (dict[Stage, float] | None): Split ratios (split mode).
        split_stratify (str | None): Column to stratify by (split mode only).
        max_samples (int | float | None): Cap dataset size.
        staged_sources (dict[Stage, list[SourceBinding]] | None): Per-stage source bindings
            (pre-split mode) — each stage's sources with their per-stage transforms.
        dataloader_options (DataLoaderOptions): DataLoader knobs shared across all stages.
        root_path (str | None): Optional prefix prepended to file-based input paths.
        cache_options (CacheOptions): In-RAM image/mask cache budget (disabled by default).
    """

    def __init__(
        self,
        target_bindings: list[TargetBinding],
        inputs_config: str | dict[str, str | dict[str, str]],
        transforms: Mapping[Stage, Transform],
        runtime: RuntimeContext,
        batch_size: int,
        seed: int = 0,
        *,
        source: DataSource | None = None,
        source_bindings: list[SourceBinding] | None = None,
        split: dict[Stage, float] | None = None,
        split_stratify: str | None = None,
        max_samples: int | float | None = None,
        staged_sources: dict[Stage, list[SourceBinding]] | None = None,
        dataloader_options: DataLoaderOptions = DataLoaderOptions(),
        root_path: str | None = None,
        cache_options: CacheOptions = CacheOptions(),
    ) -> None:
        modes = sum(value is not None for value in (source, source_bindings, staged_sources))
        if modes != 1:
            raise ValueError(
                "Provide exactly one of: source or source_bindings (split mode) or staged_sources (pre-split mode)."
            )
        if staged_sources is None and split is None:
            raise ValueError("split is required in split mode (source / source_bindings).")
        # ``source`` is the single-source shorthand for ``[SourceBinding(source, transforms)]``; both
        # forms normalize to a list of per-source bindings the split-mode setup iterates over.
        if source is not None:
            source_bindings = [SourceBinding(source=source, transforms=transforms)]
        self._source_bindings = source_bindings
        self._split = split
        self._split_stratify = split_stratify
        self._max_samples = max_samples
        self._staged_sources = staged_sources
        self._target_bindings = target_bindings
        self._inputs_config = inputs_config
        self._transforms = transforms
        self._runtime = runtime
        self._batch_size = batch_size
        self._seed = seed
        self._dataloader_options = dataloader_options
        self._root_path = root_path
        self._cache_options = cache_options
        self._cache: ArrayCache | None = None
        self._datasets: dict[Stage, list[Dataset]] = {}
        self._frames: dict[Stage, pd.DataFrame] = {}
        self._input_bindings: list[InputBinding] = []
        self._setup_done = False

    def setup(self) -> None:
        """Read data, fit encoders, resolve input bindings, and build per-stage datasets.

        Idempotent: subsequent calls are no-ops. Lightning calls this on every rank
        during ``trainer.fit()``; the first call (from ``main.py``) is the one that
        populates ``RuntimeContext`` before tasks are built.
        """
        if self._setup_done:
            return
        if self._staged_sources is not None:
            self._setup_from_staged_sources()
        else:
            self._setup_from_source_bindings()
        self._setup_done = True

    def statistics(self) -> DatasetStatistics:
        """Per-task target distributions across stages, for the pre-training report.

        Computed from the fitted encoders and the per-stage frames: each encoder
        summarizes its own column (it already knows the class vocabulary). A task whose
        encoder does not implement ``SupportsSummary`` (e.g. ``MaskEncoder``) is omitted.
        Requires ``setup`` to have run.

        Returns:
            DatasetStatistics: ``{task_name: {stage: distribution}}``.
        """
        if not self._setup_done:
            raise RuntimeError("DataModule.statistics() requires setup() to have run.")
        statistics: DatasetStatistics = {}
        for binding in self._target_bindings:
            if not isinstance(binding.encoder, SupportsSummary):
                continue
            per_stage: dict[Stage, Distribution] = {}
            for stage, frame in self._frames.items():
                distribution = binding.encoder.summarize(frame[binding.column])
                if distribution is not None:
                    per_stage[stage] = distribution
            if per_stage:
                statistics[binding.name] = per_stage
        return statistics

    def _setup_from_source_bindings(self) -> None:
        """Split mode: read each source, fit encoders on combined train, split each by ratio.

        Each source is read and split independently by the same ratios — so it is
        proportionally represented in every stage — and a stage is the ``ConcatDataset`` of
        one ``Dataset`` per source, each carrying that source's transform. A single-source /
        no-override run is one binding (identical to splitting one combined frame).
        """
        assert self._source_bindings is not None and self._split is not None, "unreachable: validated in __init__"
        source_frames = [
            _apply_max_samples(binding.source.read(), self._max_samples, self._seed)
            for binding in self._source_bindings
        ]
        # Input loaders are auto-detected from the first source's columns (all sources share them).
        self._input_bindings = _build_input_bindings(self._inputs_config, source_frames[0])
        splits = [split_dataframe(frame, self._split, self._seed, self._split_stratify) for frame in source_frames]
        # Fit encoders on the combined train data so the vocabulary spans every source.
        self._fit_encoders(pd.concat([parts[Stage.TRAIN] for parts in splits], ignore_index=True))
        stages = list(splits[0])
        self._frames = {stage: pd.concat([parts[stage] for parts in splits], ignore_index=True) for stage in stages}
        self._setup_cache(self._frames)
        for stage in stages:
            self._datasets[stage] = [
                self._build_dataset(parts[stage], binding.transforms[stage])
                for parts, binding in zip(splits, self._source_bindings, strict=True)
            ]

    def _setup_from_staged_sources(self) -> None:
        """Pre-split mode: read each stage's sources, fit encoders on combined train, one Dataset per source.

        Each stage may carry several sources, each with its own transform for that stage; a stage is
        the ``ConcatDataset`` of one ``Dataset`` per source. Encoders are fitted on the combined train
        data only (no leakage).
        """
        assert self._staged_sources is not None, "unreachable: validated in __init__"
        train_frames = [
            _apply_max_samples(binding.source.read(), self._max_samples, self._seed)
            for binding in self._staged_sources[Stage.TRAIN]
        ]
        self._input_bindings = _build_input_bindings(self._inputs_config, train_frames[0])
        self._fit_encoders(pd.concat(train_frames, ignore_index=True))
        stage_frames: dict[Stage, list[pd.DataFrame]] = {Stage.TRAIN: train_frames}
        for stage, bindings in self._staged_sources.items():
            if stage != Stage.TRAIN:
                stage_frames[stage] = [
                    _apply_max_samples(binding.source.read(), self._max_samples, self._seed) for binding in bindings
                ]
        self._frames = {stage: pd.concat(frames, ignore_index=True) for stage, frames in stage_frames.items()}
        self._setup_cache(self._frames)
        for stage, bindings in self._staged_sources.items():
            self._datasets[stage] = [
                self._build_dataset(frame, binding.transforms[stage])
                for frame, binding in zip(stage_frames[stage], bindings, strict=True)
            ]

    def _setup_cache(self, frames: dict[Stage, pd.DataFrame]) -> None:
        """Warm the cache (parent process) and wrap cacheable loaders/encoders.

        Called after encoders are fit and frames are known, before datasets are
        built. Warms from train + val paths only; wrapping is what makes the
        datasets read from the cache (``Dataset`` itself is unchanged).
        """
        if not self._cache_options.max_bytes or self._cache_options.max_bytes <= 0:
            return
        budget = self._cache_options.max_bytes
        cache = ArrayCache(budget)
        root = Path(self._root_path) if self._root_path else None
        warm_frames = [frames[stage] for stage in (Stage.TRAIN, Stage.VAL) if stage in frames]
        candidates: set[str] = set()

        for input_binding in self._input_bindings:
            if not input_binding.loader.file_based:
                continue
            keys = [resolve_path(root, value) for frame in warm_frames for value in frame[input_binding.column]]
            candidates.update(keys)
            cache.warm(keys, input_binding.loader.load, self._cache_options.workers)
        for target_binding in self._target_bindings:
            if not target_binding.encoder.file_based:
                continue
            keys = [resolve_path(root, value) for frame in warm_frames for value in frame[target_binding.column]]
            candidates.update(keys)
            cache.warm(keys, target_binding.encoder.load, self._cache_options.workers)

        self._input_bindings = [
            (
                replace(input_binding, loader=CachingLoader(input_binding.loader, cache))
                if input_binding.loader.file_based
                else input_binding
            )
            for input_binding in self._input_bindings
        ]
        self._target_bindings = [
            (
                replace(target_binding, encoder=caching_target_encoder(target_binding.encoder, cache))
                if target_binding.encoder.file_based
                else target_binding
            )
            for target_binding in self._target_bindings
        ]
        self._cache = cache
        self._log_cache_summary(cache, len(candidates), budget)

    @staticmethod
    def _log_cache_summary(cache: ArrayCache, total: int, budget: int) -> None:
        """Report how much of the dataset is in RAM vs still read from disk."""
        cached = len(cache)
        from_disk = total - cached
        log.info(
            "Data cache: %d/%d files in RAM (%.2f / %.2f GiB); %d read from disk each epoch.",
            cached,
            total,
            cache.nbytes / BYTES_PER_GIB,
            budget / BYTES_PER_GIB,
            from_disk,
        )
        if from_disk > 0:
            log.warning(
                "Cache budget (%.2f GiB) reached — only %d of %d files fit, %d will be read from disk "
                "every epoch. Raise data.cache.ram_fraction / max_gb (or free RAM) to cache more.",
                budget / BYTES_PER_GIB,
                cached,
                total,
                from_disk,
            )

    def _fit_encoders(self, frame: pd.DataFrame) -> None:
        for binding in self._target_bindings:
            if binding.column is not None:  # target-less tasks (column None) have nothing to fit
                binding.encoder.fit(frame[binding.column])
            num_classes = binding.encoder.num_classes
            if num_classes is not None:
                self._runtime.num_classes[binding.name] = num_classes

    def _build_dataset(self, frame: pd.DataFrame, transform: Transform) -> Dataset:
        return Dataset(
            frame=frame,
            input_bindings=self._input_bindings,
            target_bindings=self._target_bindings,
            transform=transform,
            root_path=self._root_path,
        )

    def _dataloader(self, stage: Stage, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        options = self._dataloader_options
        kwargs: dict[str, Any] = dict(
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=options.num_workers,
            collate_fn=collate_samples,
            pin_memory=options.pin_memory,
            drop_last=drop_last,
            persistent_workers=options.persistent_workers and options.num_workers > 0,
        )
        if options.prefetch_factor is not None and options.num_workers > 0:
            kwargs["prefetch_factor"] = options.prefetch_factor
        # User extras fill gaps; framework-owned keys (batch_size/shuffle/collate_fn/...) always win.
        # A stage is one Dataset per source (one for the common single-source case), combined here.
        return DataLoader(ConcatDataset(self._datasets[stage]), **{**options.extra_kwargs, **kwargs})

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.TRAIN, shuffle=True, drop_last=self._dataloader_options.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.VAL, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        stage = Stage.TEST if Stage.TEST in self._datasets else Stage.VAL
        return self._dataloader(stage, shuffle=False)


def _build_input_bindings(
    inputs_config: str | dict[str, str | dict[str, str]],
    frame: pd.DataFrame,
) -> list[InputBinding]:
    """Build ``InputBinding`` list from config, auto-detecting loaders from data.

    Parameters:
        inputs_config: Raw ``inputs`` value from ``DataConfig``.
        frame (pd.DataFrame): Annotation frame used for loader auto-detection.

    Returns:
        list[InputBinding]: One binding per input, with a resolved loader.
    """
    normalized = normalize_inputs(inputs_config)
    bindings: list[InputBinding] = []
    for name, spec in normalized.items():
        if isinstance(spec, str):
            column = spec
            loader_key = infer_loader_key(frame[column]) if column in frame.columns else "image"
        else:
            column = spec["column"]
            loader_key = spec.get("loader") or infer_loader_key(frame[column])
        bindings.append(InputBinding(name=name, column=column, loader=input_loaders.create(loader_key)))
    return bindings


def _apply_max_samples(frame: pd.DataFrame, max_samples: int | float | None, seed: int) -> pd.DataFrame:
    if max_samples is None:
        return frame
    if isinstance(max_samples, float):
        return frame.sample(frac=max_samples, random_state=seed).reset_index(drop=True)
    return frame.sample(n=min(max_samples, len(frame)), random_state=seed).reset_index(drop=True)
