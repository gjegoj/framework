"""The map-style Dataset: turn a DataFrame row into a model-ready ``Sample``.

Per item: load image(s) -> transform to tensors -> encode targets via codecs.
Heavy I/O (image read, mask decode) runs here, inside DataLoader workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from src.core.entities import Sample
from src.data.codecs import TargetCodec
from src.data.loaders import ImageLoader
from src.data.transforms import IMAGE_KEY, Transform


@dataclass(frozen=True)
class TargetBinding:
    """Binds a task to the data column and codec that produce its target.

    Parameters:
        name (str): Task name; also the key under which the target is stored.
        column (str): Source column in the DataFrame.
        codec (TargetCodec): Codec that decodes the raw column value.
    """

    name: str
    column: str
    codec: TargetCodec


class Dataset(TorchDataset[Sample]):
    """Map-style dataset assembling one ``Sample`` per row.

    Parameters:
        frame (pd.DataFrame): Rows for this split.
        image_column (str): Column holding the image path.
        bindings (list[TargetBinding]): Task target bindings.
        transform (Transform): Input transform (numpy image -> tensor).
        loader (ImageLoader): Image loader.
        root_path (str | None): Optional prefix prepended to image paths.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        bindings: list[TargetBinding],
        transform: Transform,
        loader: ImageLoader,
        root_path: str | None = None,
    ) -> None:
        self._frame = frame.reset_index(drop=True)
        self._image_column = image_column
        self._bindings = bindings
        self._transform = transform
        self._loader = loader
        self._root = Path(root_path) if root_path else None

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> Sample:
        row = self._frame.iloc[index]

        raw_path = str(row[self._image_column])
        path = str(self._root / raw_path) if self._root is not None else raw_path
        image = self._loader.load(path)

        sample = Sample(inputs={IMAGE_KEY: image}, meta={"index": index})
        sample = self._transform.apply(sample)
        for binding in self._bindings:
            sample.targets[binding.name] = binding.codec.encode(row[binding.column])
        return sample
