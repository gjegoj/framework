"""The map-style Dataset: turn a DataFrame row into a model-ready ``Sample``.

Per item:
1. Load all inputs via ``InputBinding.loader.load``:
   file-based loaders (images) receive a root_path-resolved path;
   raw-value loaders (text) receive the column value as-is.
2. Load targets via ``codec.load``:
   spatial codecs (masks) read the file; scalar codecs return the raw value.
3. Apply transform — inputs and spatial targets pass through together so
   geometric operations stay aligned across image and mask.
4. Finalise all targets to tensors via ``codec.to_tensor``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from src.core.entities import Sample
from src.data.bindings import InputBinding, TargetBinding
from src.transforms.input import Transform


def resolve_path(root: Path | None, raw: object) -> str:
    """Resolve a raw column value to a filesystem path, prefixing ``root`` if set."""
    text = str(raw)
    return str(root / text) if root is not None else text


class Dataset(TorchDataset[Sample]):
    """Map-style dataset assembling one ``Sample`` per row.

    Parameters:
        frame (pd.DataFrame): Rows for this split.
        input_bindings (list[InputBinding]): Per-input column + loader.
        target_bindings (list[TargetBinding]): Per-task target column + codec.
        transform (Transform): Input transform applied after loading.
        root_path (str | None): Prefix prepended to file-based input paths.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        input_bindings: list[InputBinding],
        target_bindings: list[TargetBinding],
        transform: Transform,
        root_path: str | None = None,
    ) -> None:
        self._frame = frame.reset_index(drop=True)
        self._input_bindings = input_bindings
        self._target_bindings = target_bindings
        self._transform = transform
        self._root = Path(root_path) if root_path else None

    def __len__(self) -> int:
        return len(self._frame)

    def _resolve(self, raw: object) -> str:
        return resolve_path(self._root, raw)

    def __getitem__(self, index: int) -> Sample:
        row = self._frame.iloc[index]
        sample = Sample(inputs={}, meta={"index": index})

        # 1. Load all inputs (file-based → resolved path; raw-value → as-is).
        for ib in self._input_bindings:
            value = self._resolve(row[ib.column]) if ib.loader.file_based else str(row[ib.column])
            sample.inputs[ib.name] = ib.loader.load(value)

        # 2. Load all targets (spatial → array; scalar → raw value).
        for tb in self._target_bindings:
            raw = self._resolve(row[tb.column]) if tb.codec.spatial else row[tb.column]
            sample.targets[tb.name] = tb.codec.load(raw)

        # 3. Transform: inputs + spatial targets pass through together.
        sample = self._transform.apply(sample)

        # 4. Finalise all targets to tensors.
        for tb in self._target_bindings:
            sample.targets[tb.name] = tb.codec.to_tensor(sample.targets[tb.name])

        return sample
