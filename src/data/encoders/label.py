"""Class labels: a name or index becomes its declared position; the vocabulary is never learned from rows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import ClassVar, Self

import torch
from torch import Tensor

from src.core import Distribution, TargetInfo, class_name, validate_classes
from src.data.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import counted

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

    def fit(self, values: Iterable[object]) -> Self:
        """A declared vocabulary learns nothing from the training split; it is checked against it."""
        self.validate(values)
        return self

    def position(self, label: object) -> int:
        try:
            return self._positions[str(label).strip()]
        except KeyError:
            raise LookupError(
                f"Unknown label {label!r}. Declared classes: {', '.join(self.classes.values())}."
            ) from None

    def named(self, label: object) -> str:
        """The class a cell stands for, whichever of its two spellings the table used.

        A vocabulary accepts a name and the index behind it, so a column written as indices trains and
        validates exactly like one written as words. Counting the raw cell instead would file those
        rows under ``0`` and ``1`` and report every declared class as one the data never shows — on a
        perfectly balanced column. A cell outside the vocabulary keeps its own spelling: the encoders
        refuse those when they are fitted, and a report should not be what hides the diagnosis.
        """
        try:
            return class_name(self.classes, self.position(label))
        except LookupError:
            return str(label).strip()

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

    def validate(self, values: Iterable[object]) -> None:
        self.refuse_unknown(str(value).strip() for value in values)

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        """One count per row, seeded with the vocabulary so a class nobody wrote still shows."""
        return counted(self.classes, (self.named(value) for value in values))

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

    def validate(self, values: Iterable[object]) -> None:
        self.refuse_unknown(label for value in values for label in labels_in(value, self.separator))

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        """One count per label, so the total runs past the row count wherever rows carry several."""
        return counted(self.classes, (self.named(one) for value in values for one in labels_in(value, self.separator)))

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
