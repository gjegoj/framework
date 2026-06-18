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
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.bindings import InputBinding, TargetBinding
from src.data.cache import ArrayCache, CachingLoader, CachingTargetEncoder
from src.data.collate import collate_samples
from src.data.dataset import Dataset, resolve_path
from src.data.loaders import _infer_loader_key, _normalize_inputs
from src.data.registry import input_loaders
from src.data.sources import DataSource
from src.data.split import split_dataframe
from src.transforms.sample import Transform

log = logging.getLogger(__name__)


class DataModule:
    """Builds datasets and dataloaders, and populates the RuntimeContext.

    Two construction modes (mutually exclusive):

    **Split mode** — one source, DataModule splits it by ratio::

        DataModule(source=csv_source, split={Stage.TRAIN: 0.8, ...}, ...)

    **Pre-split mode** — caller provides one ``DataSource`` per stage::

        DataModule(staged_sources={Stage.TRAIN: train_src, Stage.VAL: val_src}, ...)

    In pre-split mode encoders are fitted on train data only (no leakage).
    ``InputBinding`` loaders are auto-detected from column values at ``setup()``
    time when not explicitly specified in ``inputs_config``.

    Parameters:
        bindings (list[TargetBinding]): Per-task target column + encoder.
        inputs_config: Input column(s) and optional loader keys.
            ``str`` → single image input; ``dict`` → multiple inputs.
        transforms (dict[Stage, Transform]): Per-stage input transforms.
        runtime (RuntimeContext): Populated with num_classes and dataset sizes.
        batch_size (int): Batch size for all stages.
        seed (int): Split seed (split mode only).
        source (DataSource | None): Full annotation table (split mode).
        split (dict[Stage, float] | None): Split ratios (split mode).
        split_stratify (str | None): Column to stratify by (split mode only).
        max_samples (int | float | None): Cap dataset size.
        staged_sources (dict[Stage, DataSource] | None): Per-stage sources (pre-split mode).
        num_workers (int): DataLoader worker processes (0 → main process).
        pin_memory (bool): Pin host memory for faster CPU→GPU transfers.
        persistent_workers (bool): Keep workers alive between epochs (requires num_workers > 0).
        drop_last (bool): Drop the last incomplete batch during training.
        prefetch_factor (int | None): Batches prefetched per worker (None → PyTorch default).
        dataloader_kwargs (dict[str, Any] | None): Extra keyword args forwarded verbatim to every
            ``DataLoader`` (e.g. ``timeout``). Framework-owned keys always win over these.
        root_path (str | None): Optional prefix prepended to file-based input paths.
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
        split: dict[Stage, float] | None = None,
        split_stratify: str | None = None,
        max_samples: int | float | None = None,
        staged_sources: dict[Stage, DataSource] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        prefetch_factor: int | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        root_path: str | None = None,
        cache_bytes: int | None = None,
        cache_workers: int = 8,
    ) -> None:
        if (source is None) == (staged_sources is None):
            raise ValueError("Provide exactly one of: source+split (split mode) or staged_sources (pre-split mode).")
        if source is not None and split is None:
            raise ValueError("split is required when using source.")
        self._source = source
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
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._drop_last = drop_last
        self._prefetch_factor = prefetch_factor
        self._dataloader_kwargs = dataloader_kwargs or {}
        self._root_path = root_path
        self._cache_bytes = cache_bytes
        self._cache_workers = cache_workers
        self._cache: ArrayCache | None = None
        self._datasets: dict[Stage, Dataset] = {}
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
            self._setup_from_source()
        self._setup_done = True

    def _setup_from_source(self) -> None:
        """Split mode: read the single source, fit encoders on all data, split by ratio."""
        assert self._source is not None and self._split is not None, "unreachable: validated in __init__"
        frame = _apply_max_samples(self._source.read(), self._max_samples, self._seed)
        self._input_bindings = _build_input_bindings(self._inputs_config, frame)
        self._fit_encoders(frame)
        parts = split_dataframe(frame, self._split, self._seed, self._split_stratify)
        self._setup_cache(parts)
        for stage, part in parts.items():
            self._build_dataset(stage, part)

    def _setup_from_staged_sources(self) -> None:
        """Pre-split mode: fit encoders on train data only, build datasets from each source."""
        assert self._staged_sources is not None, "unreachable: validated in __init__"
        train_frame = _apply_max_samples(self._staged_sources[Stage.TRAIN].read(), self._max_samples, self._seed)
        self._input_bindings = _build_input_bindings(self._inputs_config, train_frame)
        self._fit_encoders(train_frame)
        frames: dict[Stage, pd.DataFrame] = {Stage.TRAIN: train_frame}
        for stage, source in self._staged_sources.items():
            if stage != Stage.TRAIN:
                frames[stage] = _apply_max_samples(source.read(), self._max_samples, self._seed)
        self._setup_cache(frames)
        for stage, frame in frames.items():
            self._build_dataset(stage, frame)

    def _setup_cache(self, frames: dict[Stage, pd.DataFrame]) -> None:
        """Warm the cache (parent process) and wrap cacheable loaders/encoders.

        Called after encoders are fit and frames are known, before datasets are
        built. Warms from train + val paths only; wrapping is what makes the
        datasets read from the cache (``Dataset`` itself is unchanged).
        """
        if not self._cache_bytes or self._cache_bytes <= 0:
            return
        budget = self._cache_bytes
        cache = ArrayCache(budget)
        root = Path(self._root_path) if self._root_path else None
        warm_frames = [frames[stage] for stage in (Stage.TRAIN, Stage.VAL) if stage in frames]
        candidates: set[str] = set()

        for input_binding in self._input_bindings:
            if not input_binding.loader.file_based:
                continue
            keys = [resolve_path(root, value) for frame in warm_frames for value in frame[input_binding.column]]
            candidates.update(keys)
            cache.warm(keys, input_binding.loader.load, self._cache_workers)
        for target_binding in self._target_bindings:
            if not target_binding.encoder.spatial:
                continue
            keys = [resolve_path(root, value) for frame in warm_frames for value in frame[target_binding.column]]
            candidates.update(keys)
            cache.warm(keys, target_binding.encoder.load, self._cache_workers)

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
                replace(target_binding, encoder=CachingTargetEncoder(target_binding.encoder, cache))
                if target_binding.encoder.spatial
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
        gib = 1024**3
        log.info(
            "Data cache: %d/%d files in RAM (%.2f / %.2f GiB); %d read from disk each epoch.",
            cached,
            total,
            cache.nbytes / gib,
            budget / gib,
            from_disk,
        )
        if from_disk > 0:
            log.warning(
                "Cache budget (%.2f GiB) reached — only %d of %d files fit, %d will be read from disk "
                "every epoch. Raise data.cache.ram_fraction / max_gb (or free RAM) to cache more.",
                budget / gib,
                cached,
                total,
                from_disk,
            )

    def _fit_encoders(self, frame: pd.DataFrame) -> None:
        for binding in self._target_bindings:
            binding.encoder.fit(frame[binding.column])
            num_classes = binding.encoder.num_classes
            if num_classes is not None:
                self._runtime.num_classes[binding.name] = num_classes

    def _build_dataset(self, stage: Stage, frame: pd.DataFrame) -> None:
        self._datasets[stage] = Dataset(
            frame=frame,
            input_bindings=self._input_bindings,
            target_bindings=self._target_bindings,
            transform=self._transforms[stage],
            root_path=self._root_path,
        )

    def _dataloader(self, stage: Stage, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        kwargs: dict = dict(
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=collate_samples,
            pin_memory=self._pin_memory,
            drop_last=drop_last,
            persistent_workers=self._persistent_workers and self._num_workers > 0,
        )
        if self._prefetch_factor is not None and self._num_workers > 0:
            kwargs["prefetch_factor"] = self._prefetch_factor
        # User extras fill gaps; framework-owned keys (batch_size/shuffle/collate_fn/...) always win.
        return DataLoader(self._datasets[stage], **{**self._dataloader_kwargs, **kwargs})

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.TRAIN, shuffle=True, drop_last=self._drop_last)

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
    normalized = _normalize_inputs(inputs_config)
    bindings: list[InputBinding] = []
    for name, spec in normalized.items():
        if isinstance(spec, str):
            column = spec
            loader_key = _infer_loader_key(frame[column]) if column in frame.columns else "image"
        else:
            column = spec["column"]
            loader_key = spec.get("loader") or _infer_loader_key(frame[column])
        bindings.append(InputBinding(name=name, column=column, loader=input_loaders.create(loader_key)))
    return bindings


def _apply_max_samples(frame: pd.DataFrame, max_samples: int | float | None, seed: int) -> pd.DataFrame:
    if max_samples is None:
        return frame
    if isinstance(max_samples, float):
        return frame.sample(frac=max_samples, random_state=seed).reset_index(drop=True)
    return frame.sample(n=min(max_samples, len(frame)), random_state=seed).reset_index(drop=True)
