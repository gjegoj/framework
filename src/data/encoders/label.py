"""Categorical targets: one class per cell, or several."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any, override

import numpy as np

from src.core.entities import Distribution
from src.core.vocabulary import ordered_names
from src.data.encoders.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import counted

log = logging.getLogger(__name__)


@target_encoder_registry.register("label")
class LabelTargetEncoder(TargetEncoder):
    """Categorical labels into class indices.

    A declared vocabulary is the contract the data is validated against —
    a typo row fails loudly instead of silently growing the class count, and
    the index space stays put when a resample drops a rare class from train.
    Undeclared, the vocabulary is learned from the training split, sorted.

    Parameters:
        classes (Mapping[int, str] | None): Declared vocabulary, index to name.
    """

    def __init__(self, classes: Mapping[int, str] | None = None) -> None:
        names = ordered_names(classes) if classes is not None else None
        self._declared = names is not None
        self._index: dict[str, int] | None = (
            {name: position for position, name in enumerate(names)} if names is not None else None
        )

    def fit(self, values: Iterable[Any]) -> None:
        if self._declared:
            assert self._index is not None
            unknown = sorted({str(value) for value in values} - self._index.keys())
            if unknown:
                known = ", ".join(self._index)
                raise LookupError(f"Values outside the declared classes: {', '.join(unknown)}. Declared: {known}.")
            return
        vocabulary = sorted({str(value) for value in values})
        self._index = {name: position for position, name in enumerate(vocabulary)}

    def encode(self, value: Any) -> int:
        if self._index is None:
            raise RuntimeError("LabelTargetEncoder is not fitted; call fit(train_values) first.")
        try:
            return self._index[str(value)]
        except KeyError:
            known = ", ".join(self._index)
            raise LookupError(f"Unknown label '{value}'. Known classes: {known}.") from None

    @property
    def num_classes(self) -> int | None:
        return len(self._index) if self._index is not None else None

    @property
    def class_names(self) -> list[str] | None:
        return list(self._index) if self._index is not None else None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per row, seeded with the vocabulary so an unused class still shows."""
        return counted(self.class_names, (str(value) for value in values))


@target_encoder_registry.register("multilabel")
class MultiLabelTargetEncoder(TargetEncoder):
    """Several labels per row into one indicator vector; the vocabulary is learned at fit.

    Cells hold a separated string (``"cat,dog"``) or a real list. A row with no labels
    encodes to all zeros: the absence of every class is itself an observation. Values are
    ``float`` because binary cross-entropy compares against probabilities.

    Parameters:
        classes (Mapping[int, str] | None): Declared vocabulary, index to name; ``None``
            learns it from the training split, sorted.
        separator (str): Separator splitting a string cell into labels.
    """

    def __init__(self, classes: Mapping[int, str] | None = None, separator: str = ",") -> None:
        if not separator:
            raise ValueError("MultiLabelTargetEncoder needs a non-empty separator.")
        self._separator = separator
        self._declared = classes is not None
        self._classes: list[str] | None = None
        self._positions: dict[str, int] = {}
        if classes is not None:
            self._adopt(ordered_names(classes))

    def _adopt(self, names: list[str]) -> None:
        """Hold the vocabulary and the index into it together, so the two cannot drift."""
        self._classes = names
        self._positions = {name: position for position, name in enumerate(names)}

    def fit(self, values: Iterable[Any]) -> None:
        if self._declared:
            assert self._classes is not None
            declared = set(self._classes)
            unknown = sorted({label for value in values for label in self._labels_in(value)} - declared)
            if unknown:
                known = ", ".join(self._classes)
                raise LookupError(f"Labels outside the declared classes: {', '.join(unknown)}. Declared: {known}.")
            return
        self._adopt(sorted({label for value in values for label in self._labels_in(value)}))

    def encode(self, value: Any) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("MultiLabelTargetEncoder is not fitted; call fit(train_values) first.")
        indicator = np.zeros(len(self._classes), dtype=np.float32)
        for label in self._labels_in(value):
            try:
                indicator[self._positions[label]] = 1.0
            except KeyError:
                known = ", ".join(self._classes)
                raise LookupError(f"Unknown label '{label}'. Known classes: {known}.") from None
        return indicator

    def _labels_in(self, value: Any) -> set[str]:
        """The labels a cell carries, in either of the two forms a table stores them."""
        if isinstance(value, list | tuple | set):
            return {str(item).strip() for item in value if str(item).strip()}
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return set()
        return {part.strip() for part in str(value).split(self._separator) if part.strip()}

    @property
    def num_classes(self) -> int | None:
        return len(self._classes) if self._classes is not None else None

    @property
    def class_names(self) -> list[str] | None:
        return list(self._classes) if self._classes is not None else None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per label, so the total exceeds the row count wherever rows carry several."""
        return counted(self.class_names, (label for value in values for label in self._labels_in(value)))
