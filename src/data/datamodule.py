"""DataModule: framework-agnostic orchestration of the data pipeline.

Deliberately not a ``LightningDataModule`` — it is plain and unit-testable. The
thin Lightning wrapper that delegates here is added in the training layer.

On ``setup`` it: reads the source, fits target codecs (inferring ``num_classes``
into the RuntimeContext), resolves ``InputBinding``s with loader auto-detection,
splits into per-stage datasets, and records dataset sizes.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from torch.utils.data import DataLoader

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.bindings import InputBinding, TargetBinding
from src.data.collate import collate_samples
from src.data.dataset import Dataset
from src.data.loaders import _infer_loader_key, _normalize_inputs, input_loaders
from src.data.sources import DataSource
from src.data.split import split_dataframe
from src.data.transforms import Transform


class DataModule:
    """Builds datasets and dataloaders, and populates the RuntimeContext.

    Two construction modes (mutually exclusive):

    **Split mode** — one source, DataModule splits it by ratio::

        DataModule(source=csv_source, split={Stage.TRAIN: 0.8, ...}, ...)

    **Pre-split mode** — caller provides one ``DataSource`` per stage::

        DataModule(staged_sources={Stage.TRAIN: train_src, Stage.VAL: val_src}, ...)

    In pre-split mode codecs are fitted on train data only (no leakage).
    ``InputBinding`` loaders are auto-detected from column values at ``setup()``
    time when not explicitly specified in ``inputs_config``.

    Parameters:
        bindings (list[TargetBinding]): Per-task target column + codec.
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
        num_workers (int): DataLoader worker processes.
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
        root_path: str | None = None,
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
        self._root_path = root_path
        self._datasets: dict[Stage, Dataset] = {}
        self._input_bindings: list[InputBinding] = []

    def setup(self) -> None:
        """Read data, fit codecs, resolve input bindings, and build per-stage datasets."""
        if self._staged_sources is not None:
            self._setup_from_staged_sources()
        else:
            self._setup_from_source()

    def _setup_from_source(self) -> None:
        """Split mode: read the single source, fit codecs on all data, split by ratio."""
        assert self._source is not None and self._split is not None, "unreachable: validated in __init__"
        frame = _apply_max_samples(self._source.read(), self._max_samples, self._seed)
        self._input_bindings = _build_input_bindings(self._inputs_config, frame)
        self._fit_codecs(frame)
        for stage, part in split_dataframe(frame, self._split, self._seed, self._split_stratify).items():
            self._build_dataset(stage, part)

    def _setup_from_staged_sources(self) -> None:
        """Pre-split mode: fit codecs on train data only, build datasets from each source."""
        assert self._staged_sources is not None, "unreachable: validated in __init__"
        train_frame = _apply_max_samples(self._staged_sources[Stage.TRAIN].read(), self._max_samples, self._seed)
        self._input_bindings = _build_input_bindings(self._inputs_config, train_frame)
        self._fit_codecs(train_frame)
        self._build_dataset(Stage.TRAIN, train_frame)
        for stage, source in self._staged_sources.items():
            if stage != Stage.TRAIN:
                self._build_dataset(stage, _apply_max_samples(source.read(), self._max_samples, self._seed))

    def _fit_codecs(self, frame: pd.DataFrame) -> None:
        for binding in self._target_bindings:
            binding.codec.fit(frame[binding.column])
            num_classes = binding.codec.num_classes
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
        self._runtime.dataset_sizes[stage] = len(self._datasets[stage])

    def _dataloader(self, stage: Stage, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            self._datasets[stage],
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=collate_samples,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.TRAIN, shuffle=True, drop_last=True)

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
