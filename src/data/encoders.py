"""Target encoders (data-layer I/O): encode a raw target value into a tensor.

All encoders follow the same two-step interface:
  1. ``load(value)``    — pre-transform: return a representation that can ride
                          through the transform pipeline. For scalar encoders this
                          is an identity (returns the raw column value); for
                          spatial encoders (masks) it reads the file into an array.
  2. ``to_tensor(val)`` — post-transform: convert whatever ``load`` (and the
                          transform) produced into a final model-ready tensor.
                          For scalar encoders this is where the encoding happens
                          (label lookup, float cast, etc.).

Keeping both steps uniform means ``Dataset.__getitem__`` has a single clean
loop for each stage instead of branching on encoder type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from src.data.registry import target_encoders


class TargetEncoder(ABC):
    """Two-step encoder: ``load`` (pre-transform) → ``to_tensor`` (post-transform).

    ``spatial`` marks encoders whose ``load`` returns a numpy array (a mask) that
    must ride through the same geometric transform as the image. Scalar encoders
    have ``spatial = False`` and their ``load`` is a no-op identity.
    """

    spatial: bool = False

    @abstractmethod
    def fit(self, values: Iterable[Any]) -> None:
        """Learn any state needed to encode (e.g. the class vocabulary)."""

    @abstractmethod
    def load(self, value: Any) -> Any:
        """Pre-transform step.

        Scalar encoders: return ``value`` unchanged (identity).
        Spatial encoders: read the file at ``value`` into a raw numpy array.
        """

    @abstractmethod
    def to_tensor(self, value: Any) -> Tensor:
        """Post-transform step: convert to a final model-ready tensor.

        Scalar encoders: do the full encoding here (label lookup, float cast, ...).
        Spatial encoders: fix the dtype of the (already-transformed) array/tensor.
        """

    @property
    def num_classes(self) -> int | None:
        """Number of classes if categorical, else ``None``."""
        return None


class _CategoricalEncoder(TargetEncoder):
    """Shared vocabulary handling for label encoders: the index⇄label maps + validation.

    Subclasses supply only the encoding (``to_tensor``) and pick which labels each row
    contributes to ``fit`` (a single label, or many for multi-label). ``load`` is identity —
    the raw string rides through the transform pipeline unchanged.

    Parameters:
        class_mapping (dict[int, str] | None): Fixed index→label map; if provided ``fit`` only
            validates, and the vocabulary is not inferred from data.
    """

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        if class_mapping is not None:
            self._set_mapping([class_mapping[i] for i in sorted(class_mapping)])

    def _set_mapping(self, labels: list[str]) -> None:
        self._index_to_label = dict(enumerate(labels))
        self._label_to_index = {label: index for index, label in self._index_to_label.items()}

    def _require_mapping(self) -> None:
        if not self._label_to_index:
            raise ValueError(
                f"{type(self).__name__} requires 'class_mapping' to be provided explicitly. "
                "Set 'class_mapping' in TaskConfig or pass it to the encoder constructor."
            )

    def _check_known(self, labels: Iterable[str]) -> None:
        unknown = set(labels) - set(self._label_to_index)
        if unknown:
            raise ValueError(
                f"Column contains labels not in class_mapping: {sorted(unknown)}. "
                f"Known: {sorted(self._label_to_index)}."
            )

    def _index_of(self, label: str) -> int:
        try:
            return self._label_to_index[label]
        except KeyError as error:
            raise KeyError(f"Unknown label {label!r}. Known labels: {sorted(self._label_to_index)}.") from error

    def load(self, value: Any) -> Any:
        return value  # identity — the raw label string passes through the transform

    @property
    def num_classes(self) -> int | None:
        return len(self._index_to_label) or None

    @property
    def class_mapping(self) -> dict[int, str]:
        return dict(self._index_to_label)


@target_encoders.register("label")
class LabelEncoder(_CategoricalEncoder):
    """Maps a single categorical label to its integer class index (multiclass/binary)."""

    def fit(self, values: Iterable[Any]) -> None:
        self._require_mapping()
        self._check_known(str(value) for value in values)

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(self._index_of(str(value)), dtype=torch.long)


@target_encoders.register("multilabel")
class MultiLabelEncoder(_CategoricalEncoder):
    """Maps a delimited label string to a multi-hot ``[C]`` float tensor.

    Parameters:
        separator (str): Delimiter used to split the label string (default ``","``).
        class_mapping (dict[int, str] | None): Fixed vocabulary; ``fit`` only validates.
    """

    def __init__(self, separator: str = ",", class_mapping: dict[int, str] | None = None) -> None:
        super().__init__(class_mapping)
        self._separator = separator

    def _split(self, value: Any) -> list[str]:
        return [part.strip() for part in str(value).split(self._separator) if part.strip()]

    def fit(self, values: Iterable[Any]) -> None:
        self._require_mapping()
        self._check_known(label for value in values for label in self._split(value))

    def to_tensor(self, value: Any) -> Tensor:
        multi_hot = torch.zeros(len(self._index_to_label), dtype=torch.float)
        for label in self._split(value):
            multi_hot[self._index_of(label)] = 1.0
        return multi_hot


@target_encoders.register("scalar")
class ScalarEncoder(TargetEncoder):
    """Encodes a scalar numeric target as a ``[]`` float tensor (regression)."""

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> Any:
        return value  # identity

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(float(value), dtype=torch.float)


@target_encoders.register("mask")
class MaskEncoder(TargetEncoder):
    """Spatial encoder for index masks: a single-channel PNG of class indices.

    ``load`` reads the PNG into a ``[H, W]`` uint8 array before the transform so
    Albumentations can resize/flip it together with the image. ``to_tensor``
    casts the result to a ``[H, W]`` long tensor for the criterion.

    When ``class_mapping`` is provided, ``num_classes`` is inferred from it —
    consistent with the categorical encoders. Otherwise ``num_classes`` stays ``None``
    and the task config must supply an explicit ``num_classes``.

    Parameters:
        class_mapping (dict[int, str] | None): Index → label map, e.g.
            ``{0: "background", 1: "defect"}``. Determines class count.
    """

    spatial = True

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._num_classes: int | None = len(class_mapping) if class_mapping is not None else None

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

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
