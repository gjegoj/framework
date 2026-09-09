"""The arithmetic behind the report a run prints before its first epoch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.taxonomy import Stage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ClassDistribution:
    """How many of each class a column holds — the imbalance, before it surprises anyone.

    Zero-count classes are kept: a class the training split never shows is the most useful
    line. ``counts`` sums to the row count for a single-label column, to more for a
    multilabel one, and to pixels for a mask.
    """

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def shares(self) -> dict[str, float]:
        """Each class as a fraction of the total; all zero when there is nothing to divide."""
        total = self.total
        return {name: (count / total if total else 0.0) for name, count in self.counts.items()}


@dataclass(frozen=True, slots=True)
class ValueDistribution:
    """The five-number summary of a numeric column, plus its mean and deviation.

    Quantiles rather than a histogram: the shape of a target is read from where its
    mass sits, and the quartiles say that in five numbers that fit a terminal row —
    where a histogram would need a bin count nobody has a principled value for.
    """

    count: int
    mean: float
    deviation: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float


type Distribution = ClassDistribution | ValueDistribution
"""What one target column looks like, in whichever of the two shapes fits it."""


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """What a run is about to train on: how much of it there is, and what it holds.

    Row counts are here because a split that went wrong — an empty stage, a test set larger
    than train — shows up there and nowhere else.
    """

    rows: dict[Stage, int] = field(default_factory=dict)
    targets: dict[str, dict[Stage, Distribution]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether there is anything at all to report."""
        return bool(self.rows or self.targets)


@dataclass(frozen=True, slots=True)
class Bars:
    """Named quantities drawn as grouped bars — a class balance across stages.

    One series per group and one value per label within it, so a class missing from one
    split is a gap rather than a number to hunt for.
    """

    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]
    xaxis: str
    yaxis: str


@dataclass(frozen=True, slots=True)
class BoxPlot:
    """Five-number summaries drawn as boxes — one per series, on shared axes.

    Carries the ``ValueDistribution``s themselves, not a copy of their numbers. Whiskers are
    the observed minimum and maximum, not Tukey's fences: outliers would need the raw values
    held in memory for a picture drawn once.
    """

    series: tuple[str, ...]
    boxes: tuple[ValueDistribution, ...]
    xaxis: str
    yaxis: str


def counted(names: Sequence[str] | None, labels: Iterable[str]) -> ClassDistribution:
    """Count labels, starting from every declared class so the unused ones still show.

    One of the two shapes a target's statistics take — *counted* (classes) against
    *measured* (numbers); the encoder picks. A class the split never produced is the most
    useful row, so the vocabulary seeds the count; a label outside it is still counted,
    because the encoders refuse those at ``fit`` and a report should not hide the diagnosis.

    Parameters:
        names (Sequence[str] | None): The declared vocabulary, seeded at zero.
        labels (Iterable[str]): One label per sample, or per pixel for a mask.
    """
    counts = dict.fromkeys(names or (), 0)
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return ClassDistribution(counts=counts)


def measured(values: Iterable[Any]) -> ValueDistribution | None:
    """The five-number summary of a numeric column, or ``None`` when it holds no number.

    The other shape beside ``counted`` — a target with a spread rather than a
    vocabulary: regression, and the binned encoders.

    ``NaN`` is dropped rather than propagated: one missing cell would otherwise turn
    every statistic into ``nan`` and the row would say nothing at all, where ``count``
    against the stage's row count already shows how much is missing.
    """
    numbers = np.asarray([float(value) for value in values], dtype=float)
    numbers = numbers[~np.isnan(numbers)]
    if numbers.size == 0:
        return None
    minimum, q25, median, q75, maximum = (float(edge) for edge in np.percentile(numbers, [0, 25, 50, 75, 100]))
    return ValueDistribution(
        count=int(numbers.size),
        mean=float(numbers.mean()),
        # Sample deviation, and 0.0 for a single value rather than the nan numpy gives.
        deviation=float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0,
        minimum=minimum,
        q25=q25,
        median=median,
        q75=q75,
        maximum=maximum,
    )
