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
