"""The map-style Dataset: turn a DataFrame row into a model-ready ``Sample``.

Per item:
1. Load all inputs via ``InputBinding.loader.load``:
   file-based loaders (images) receive a root_path-resolved path;
   raw-value loaders (text) receive the column value as-is.
2. Load targets via ``encoder.load``:
   spatial encoders (masks) read the file; scalar encoders return the raw value.
3. Apply transform — inputs and spatial targets pass through together so
   geometric operations stay aligned across image and mask.
4. Finalise all targets to tensors via ``encoder.to_tensor``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from src.core.entities import Sample, SampleMeta
from src.data.bindings import InputBinding, TargetBinding
from src.transforms.sample import Transform


def resolve_path(root: Path | None, raw: object) -> str:
    """Resolve a raw column value to a filesystem path, prefixing ``root`` if set."""
    text = str(raw)
    return str(root / text) if root is not None else text


class Dataset(TorchDataset[Sample]):
    """Map-style dataset assembling one ``Sample`` per row.

    Parameters:
        frame (pd.DataFrame): Rows for this split.
        input_bindings (list[InputBinding]): Per-input column + loader.
        target_bindings (list[TargetBinding]): Per-task target column + encoder.
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
        # ``input_sources``/``target_sources`` record the resolved source path of each
        # file-based input / spatial (mask) target, keyed by binding name — so a prediction
        # can be traced back to its file (collate transposes these to per-sample lists).
        sample = Sample(inputs={}, meta=SampleMeta(index=index, input_sources={}, target_sources={}))

        # 1. Load all inputs (file-based → resolved path, also kept as a source; raw-value → as-is).
        for input_binding in self._input_bindings:
            if input_binding.loader.file_based:
                value = self._resolve(row[input_binding.column])
                sample.meta["input_sources"][input_binding.name] = value
            else:
                value = str(row[input_binding.column])
            sample.inputs[input_binding.name] = input_binding.loader.load(value)

        # 2. Load all targets (spatial → resolved path, also kept as a source; scalar → raw value).
        for target_binding in self._target_bindings:
            column_value = row[target_binding.column]
            if target_binding.encoder.spatial:
                raw = self._resolve(column_value)
                sample.meta["target_sources"][target_binding.name] = raw
            else:
                raw = column_value
            sample.targets[target_binding.name] = target_binding.encoder.load(raw)

        # 3. Transform: inputs + spatial targets pass through together.
        sample = self._transform.apply(sample)

        # 4. Finalise all targets to tensors.
        for target_binding in self._target_bindings:
            current = sample.targets[target_binding.name]
            sample.targets[target_binding.name] = target_binding.encoder.to_tensor(current)

        return sample
