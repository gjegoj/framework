"""Target codecs (data-layer I/O): decode a raw target value into a tensor.

All codecs follow the same two-step interface:
  1. ``load(value)``    — pre-transform: return a representation that can ride
                          through the transform pipeline. For scalar codecs this
                          is an identity (returns the raw column value); for
                          spatial codecs (masks) it reads the file into an array.
  2. ``to_tensor(val)`` — post-transform: convert whatever ``load`` (and the
                          transform) produced into a final model-ready tensor.
                          For scalar codecs this is where the encoding happens
                          (label lookup, float cast, etc.).

Keeping both steps uniform means ``Dataset.__getitem__`` has a single clean
loop for each stage instead of branching on codec type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from src.core.registry import Registry

target_codecs: Registry[TargetCodec] = Registry("target_codec")


class TargetCodec(ABC):
    """Two-step codec: ``load`` (pre-transform) → ``to_tensor`` (post-transform).

    ``spatial`` marks codecs whose ``load`` returns a numpy array (a mask) that
    must ride through the same geometric transform as the image. Scalar codecs
    have ``spatial = False`` and their ``load`` is a no-op identity.
    """

    spatial: bool = False

    @abstractmethod
    def fit(self, values: Iterable[Any]) -> None:
        """Learn any state needed to encode (e.g. the class vocabulary)."""

    @abstractmethod
    def load(self, value: Any) -> Any:
        """Pre-transform step.

        Scalar codecs: return ``value`` unchanged (identity).
        Spatial codecs: read the file at ``value`` into a raw numpy array.
        """

    @abstractmethod
    def to_tensor(self, value: Any) -> Tensor:
        """Post-transform step: convert to a final model-ready tensor.

        Scalar codecs: do the full encoding here (label lookup, float cast, ...).
        Spatial codecs: fix the dtype of the (already-transformed) array/tensor.
        """

    @property
    def num_classes(self) -> int | None:
        """Number of classes if categorical, else ``None``."""
        return None


@target_codecs.register("label_index")
class LabelIndexCodec(TargetCodec):
    """Maps categorical labels to integer class indices (multiclass/binary).

    Parameters:
        class_mapping (dict[int, str] | None): Fixed index->label map; if provided
            ``fit`` is a no-op and the vocabulary is not inferred from data.
    """

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        if class_mapping is not None:
            self._set_mapping([class_mapping[i] for i in sorted(class_mapping)])

    def _set_mapping(self, labels: list[str]) -> None:
        self._index_to_label = dict(enumerate(labels))
        self._label_to_index = {label: idx for idx, label in self._index_to_label.items()}

    def fit(self, values: Iterable[Any]) -> None:
        if not self._label_to_index:
            raise ValueError(
                "LabelIndexCodec requires 'class_mapping' to be provided explicitly. "
                "Set 'class_mapping' in TaskConfig or pass it to the codec constructor."
            )
        unknown = {str(v) for v in values} - set(self._label_to_index)
        if unknown:
            raise ValueError(
                f"Column contains labels not in class_mapping: {sorted(unknown)}. "
                f"Known: {sorted(self._label_to_index)}."
            )

    def load(self, value: Any) -> Any:
        return value  # identity — raw label string passes through the transform

    def to_tensor(self, value: Any) -> Tensor:
        try:
            index = self._label_to_index[str(value)]
        except KeyError as error:
            known = sorted(self._label_to_index)
            raise KeyError(f"Unknown label {value!r}. Known labels: {known}.") from error
        return torch.tensor(index, dtype=torch.long)

    @property
    def num_classes(self) -> int | None:
        return len(self._index_to_label) or None

    @property
    def class_mapping(self) -> dict[int, str]:
        return dict(self._index_to_label)


@target_codecs.register("multilabel_binarize")
class MultiLabelBinarizeCodec(TargetCodec):
    """Maps a delimited label string to a multi-hot ``[C]`` float tensor.

    Parameters:
        separator (str): Delimiter used to split the label string (default ``","``).
        class_mapping (dict[int, str] | None): Fixed vocabulary; ``fit`` is no-op.
    """

    def __init__(self, separator: str = ",", class_mapping: dict[int, str] | None = None) -> None:
        self._separator = separator
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        if class_mapping is not None:
            self._set_mapping([class_mapping[i] for i in sorted(class_mapping)])

    def _set_mapping(self, labels: list[str]) -> None:
        self._index_to_label = dict(enumerate(labels))
        self._label_to_index = {label: idx for idx, label in self._index_to_label.items()}

    def _split(self, value: Any) -> list[str]:
        return [part.strip() for part in str(value).split(self._separator) if part.strip()]

    def fit(self, values: Iterable[Any]) -> None:
        if not self._label_to_index:
            raise ValueError(
                "MultiLabelBinarizeCodec requires 'class_mapping' to be provided explicitly. "
                "Set 'class_mapping' in TaskConfig or pass it to the codec constructor."
            )
        unknown: set[str] = set()
        for v in values:
            unknown.update(set(self._split(v)) - set(self._label_to_index))
        if unknown:
            raise ValueError(
                f"Column contains labels not in class_mapping: {sorted(unknown)}. "
                f"Known: {sorted(self._label_to_index)}."
            )

    def load(self, value: Any) -> Any:
        return value  # identity

    def to_tensor(self, value: Any) -> Tensor:
        vec = torch.zeros(len(self._index_to_label), dtype=torch.float)
        for label in self._split(value):
            try:
                vec[self._label_to_index[label]] = 1.0
            except KeyError as error:
                known = sorted(self._label_to_index)
                raise KeyError(f"Unknown label {label!r}. Known labels: {known}.") from error
        return vec

    @property
    def num_classes(self) -> int | None:
        return len(self._index_to_label) or None

    @property
    def class_mapping(self) -> dict[int, str]:
        return dict(self._index_to_label)


@target_codecs.register("float")
class FloatCodec(TargetCodec):
    """Encodes a scalar numeric target as a ``[]`` float tensor (regression)."""

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> Any:
        return value  # identity

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(float(value), dtype=torch.float)


@target_codecs.register("mask")
class MaskCodec(TargetCodec):
    """Spatial codec for index masks: a single-channel PNG of class indices.

    ``load`` reads the PNG into a ``[H, W]`` uint8 array before the transform so
    Albumentations can resize/flip it together with the image. ``to_tensor``
    casts the result to a ``[H, W]`` long tensor for the criterion.

    ``num_classes`` is ``None`` — segmentation tasks declare the class count in
    config; scanning every mask file would be needlessly expensive.
    """

    spatial = True

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> np.ndarray:
        mask = cv2.imread(str(value), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {value}")
        return mask

    def to_tensor(self, value: Any) -> Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
        return tensor.long()
