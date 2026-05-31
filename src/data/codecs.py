"""Target codecs (data-layer I/O): decode a raw target value into a tensor.

Per the split-codec design, this layer does the heavy/format-specific decoding
in DataLoader workers (e.g. label string -> class index). The lighter
shape-adaptation for loss/metrics lives in the task layer (added later).

Codecs are ``fit`` on the data once (to infer e.g. the class set), then
``encode`` each raw value. ``num_classes`` lets the DataModule populate the
RuntimeContext so heads can be sized from data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from src.core.registry import Registry

target_codecs: Registry[TargetCodec] = Registry("target_codec")


class TargetCodec(ABC):
    """Decodes raw target values from a data row into model-ready tensors."""

    @abstractmethod
    def fit(self, values: Iterable[Any]) -> None:
        """Learn any state needed to encode (e.g. the class vocabulary)."""

    @abstractmethod
    def encode(self, value: Any) -> Tensor:
        """Encode a single raw target value into a tensor."""

    @property
    def num_classes(self) -> int | None:
        """Number of classes if categorical, else ``None`` (e.g. regression)."""
        return None


@target_codecs.register("label_index")
class LabelIndexCodec(TargetCodec):
    """Maps categorical labels to integer class indices (multiclass/binary).

    The class vocabulary is the sorted set of observed labels, giving a stable,
    reproducible index assignment.

    Parameters:
        class_mapping (dict[int, str] | None): Optional fixed index->label map;
            if provided, ``fit`` is a no-op and the vocabulary is not inferred.
    """

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        if class_mapping is not None:
            self._set_mapping([class_mapping[index] for index in sorted(class_mapping)])

    def _set_mapping(self, labels: list[str]) -> None:
        self._index_to_label = dict(enumerate(labels))
        self._label_to_index = {label: index for index, label in self._index_to_label.items()}

    def fit(self, values: Iterable[Any]) -> None:
        if self._label_to_index:  # fixed mapping was provided
            return
        labels = sorted({str(value) for value in values})
        self._set_mapping(labels)

    def encode(self, value: Any) -> Tensor:
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
        """Return the inferred ``index -> label`` mapping."""
        return dict(self._index_to_label)


@target_codecs.register("multilabel_binarize")
class MultiLabelBinarizeCodec(TargetCodec):
    """Maps a delimited string of labels to a multi-hot float tensor.

    Fits a sorted class vocabulary from all observed labels, then encodes each
    value as a ``[C]`` float tensor (1.0 where the class is present).

    Parameters:
        separator (str): Delimiter used to split the label string (default ``","``).
        class_mapping (dict[int, str] | None): Optional fixed index->label map;
            if provided, ``fit`` is a no-op.
    """

    def __init__(
        self,
        separator: str = ",",
        class_mapping: dict[int, str] | None = None,
    ) -> None:
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
        if self._label_to_index:
            return
        all_labels: set[str] = set()
        for value in values:
            all_labels.update(self._split(value))
        self._set_mapping(sorted(all_labels))

    def encode(self, value: Any) -> Tensor:
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
        """Return the inferred ``index -> label`` mapping."""
        return dict(self._index_to_label)


@target_codecs.register("float")
class FloatCodec(TargetCodec):
    """Encodes a scalar numeric target as a ``[]`` float tensor (regression).

    ``num_classes`` is always ``None`` — no class vocabulary to infer.
    """

    def fit(self, values: Iterable[Any]) -> None:
        pass  # nothing to learn for a continuous target

    def encode(self, value: Any) -> Tensor:
        return torch.tensor(float(value), dtype=torch.float)
