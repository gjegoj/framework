"""The arithmetic behind the report a run prints before its first epoch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.entities import ClassDistribution, ValueDistribution

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def counted(names: Sequence[str] | None, labels: Iterable[str]) -> ClassDistribution:
    """Count labels, starting from every declared class so the unused ones still show.

    One of the two shapes every target here takes — *counted*, how many samples or
    pixels carry each class (classification, multilabel, segmentation), against
    *measured* for a column of numbers. Its encoder picks, since only the encoder
    knows the vocabulary and the parsing.

    Seeding with the vocabulary is the whole point: a class the split never produced is
    the most useful row in the table, and counting only what appeared would quietly
    leave it out. A label outside the vocabulary is still counted — the encoders refuse
    those at ``fit``, and a report that hid one would be hiding the diagnosis.

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
