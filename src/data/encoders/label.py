"""Categorical targets: one class per cell, or several."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any, override

import numpy as np

from src.data.encoders.base import VocabularyTargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import Distribution, counted

log = logging.getLogger(__name__)


@target_encoder_registry.register("label")
class LabelTargetEncoder(VocabularyTargetEncoder):
    """Categorical labels into class indices, against a declared vocabulary.

    The vocabulary is the contract the data is validated against: a typo row fails at
    ``fit`` instead of growing the class count, and the index space stays put whatever
    rows a split or a sample cap left in train.

    Parameters:
        classes (Mapping[int, str]): The vocabulary, index to name.
    """

    def __init__(self, classes: Mapping[int, str]) -> None:
        super().__init__(classes)

    @override
    def validate(self, values: Iterable[Any]) -> None:
        unknown = sorted({str(value) for value in values} - self._positions.keys())
        if unknown:
            known = ", ".join(self._names)
            raise LookupError(f"Values outside the declared classes: {', '.join(unknown)}. Declared: {known}.")

    def encode(self, value: Any) -> int:
        try:
            return self._positions[str(value)]
        except KeyError:
            known = ", ".join(self._names)
            raise LookupError(f"Unknown label '{value}'. Known classes: {known}.") from None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per row, seeded with the vocabulary so an unused class still shows."""
        return counted(self.class_names, (str(value) for value in values))


@target_encoder_registry.register("multilabel")
class MultiLabelTargetEncoder(VocabularyTargetEncoder):
    """Several labels per row into one indicator vector, against a declared vocabulary.

    Cells hold a separated string (``"cat,dog"``) or a real list. A row with no labels
    encodes to all zeros: the absence of every class is itself an observation. Values are
    ``float`` because binary cross-entropy compares against probabilities.

    Parameters:
        classes (Mapping[int, str]): The vocabulary, index to name.
        separator (str): Separator splitting a string cell into labels.
    """

    def __init__(self, classes: Mapping[int, str], separator: str = ",") -> None:
        if not separator:
            raise ValueError("MultiLabelTargetEncoder needs a non-empty separator.")
        self._separator = separator
        super().__init__(classes)

    @override
    def validate(self, values: Iterable[Any]) -> None:
        unknown = sorted(
            {label for value in values for label in labels_in(value, self._separator)} - self._positions.keys()
        )
        if unknown:
            known = ", ".join(self._names)
            raise LookupError(f"Labels outside the declared classes: {', '.join(unknown)}. Declared: {known}.")

    def encode(self, value: Any) -> np.ndarray:
        indicator = np.zeros(len(self._names), dtype=np.float32)
        for label in labels_in(value, self._separator):
            try:
                indicator[self._positions[label]] = 1.0
            except KeyError:
                known = ", ".join(self._names)
                raise LookupError(f"Unknown label '{label}'. Known classes: {known}.") from None
        return indicator

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per label, so the total exceeds the row count wherever rows carry several."""
        return counted(self.class_names, (label for value in values for label in labels_in(value, self._separator)))


def labels_in(value: Any, separator: str) -> set[str]:
    """The labels one multilabel cell carries, in either of the two forms a table stores them.

    Shared with the stratified split, so a row is read the same way when it is divided
    and when it is encoded.
    """
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {part.strip() for part in str(value).split(separator) if part.strip()}
