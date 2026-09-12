"""The arithmetic behind the report a run can print before its first epoch.

Two shapes and two functions, because a target column is either a vocabulary or a spread. Which of
them a column is, is not decided here: the encoder that reads the cells is the one thing that knows
what they hold, so it is the one that answers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from src.core import ClassDistribution, ValueDistribution


def counted(classes: Mapping[int, str] | None, labels: Iterable[str]) -> ClassDistribution:
    """Count labels, starting from every declared class so the unused ones still show.

    The vocabulary seeds the count, because a class the split never produced is the most useful row on
    the table. A label outside it is still counted rather than dropped: the encoders refuse those when
    they are fitted, and a report should not be the thing that hides the diagnosis.

    Parameters:
        classes: The declared vocabulary, seeded at zero; ``None`` counts only what is there.
        labels: One label per sample, or per pixel for a mask.
    """
    counts = dict.fromkeys((classes or {}).values(), 0)
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return ClassDistribution(counts=counts)


def _numbers(values: Iterable[object]) -> Iterator[float]:
    """Every cell that is a number, and only those."""
    for value in values:
        try:
            yield float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue


def measured(values: Iterable[object]) -> ValueDistribution | None:
    """The five-number summary of a numeric column, or ``None`` where it holds no number at all.

    A cell that holds no number — a blank, an ``n/a``, a stray word — is dropped rather than carried.
    One of them would otherwise turn every statistic into ``nan``, or end the run from inside a
    display with an unnamed conversion error; the count against the split's row count already says
    how many were missing, which is the honest reading of a column with a gap in it.
    """
    numbers = np.asarray(list(_numbers(values)), dtype=np.float64)
    numbers = numbers[~np.isnan(numbers)]
    if numbers.size == 0:
        return None
    minimum, q25, median, q75, maximum = (float(edge) for edge in np.percentile(numbers, [0, 25, 50, 75, 100]))
    return ValueDistribution(
        count=int(numbers.size),
        mean=float(numbers.mean()),
        # The sample deviation, and zero for a single value rather than the nan numpy answers with.
        deviation=float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0,
        minimum=minimum,
        q25=q25,
        median=median,
        q75=q75,
        maximum=maximum,
    )
