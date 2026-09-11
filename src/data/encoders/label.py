"""Class labels: a name or index becomes its declared position; the vocabulary is never learned from rows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import ClassVar

import torch
from torch import Tensor

from src.core import TargetInfo, validate_classes
from src.data.base import TargetEncoder
from src.data.registry import target_encoder_registry

SEPARATOR = ","
"""How a cell lists several labels; the splitter reads a column the same way when it stratifies by one."""


class VocabularyEncoder(TargetEncoder):
    """Shared by encoders built with a declared ``classes`` mapping."""

    takes_classes: ClassVar[bool] = True

    def __init__(self, *, classes: Mapping[int, str]) -> None:
        validate_classes(classes)
        self.classes = dict(classes)
        self._positions = {name: index for index, name in classes.items()} | {str(index): index for index in classes}

    @property
    def info(self) -> TargetInfo:
        return TargetInfo(classes=self.classes)

    def position(self, label: object) -> int:
        try:
            return self._positions[str(label).strip()]
        except KeyError:
            raise LookupError(
                f"Unknown label {label!r}. Declared classes: {', '.join(self.classes.values())}."
            ) from None

    def refuse_unknown(self, labels: Iterable[str]) -> None:
        unknown = sorted(set(labels) - self._positions.keys())
        if unknown:
            raise LookupError(
                f"Values outside the declared classes: {', '.join(unknown)}. "
                f"Declared: {', '.join(self.classes.values())}."
            )


@target_encoder_registry.register("label")
class LabelEncoder(VocabularyEncoder):
    """One class per sample, as an index tensor."""

    def fit(self, values: Iterable[object]) -> LabelEncoder:
        self.validate(values)
        return self

    def validate(self, values: Iterable[object]) -> None:
        self.refuse_unknown(str(value).strip() for value in values)

    def encode(self, value: object) -> Tensor:
        return torch.tensor(self.position(value), dtype=torch.long)


@target_encoder_registry.register("multilabel")
class MultilabelEncoder(VocabularyEncoder):
    """Any number of classes per sample, as a float indicator vector; an empty cell is a negative."""

    def __init__(self, *, classes: Mapping[int, str], separator: str = SEPARATOR) -> None:
        if not separator:
            raise ValueError("multilabel needs a non-empty separator.")
        super().__init__(classes=classes)
        self.separator = separator

    def fit(self, values: Iterable[object]) -> MultilabelEncoder:
        self.validate(values)
        return self

    def validate(self, values: Iterable[object]) -> None:
        self.refuse_unknown(label for value in values for label in labels_in(value, self.separator))

    def encode(self, value: object) -> Tensor:
        indicator = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in labels_in(value, self.separator):
            indicator[self.position(label)] = 1.0
        return indicator


def labels_in(value: object, separator: str = SEPARATOR) -> set[str]:
    """The labels a cell carries: a list, a separated string, or nothing at all."""
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {part.strip() for part in str(value).split(separator) if part.strip()}
