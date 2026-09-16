"""The arithmetic behind the report a run can print before its first epoch.

Two shapes and two functions, because a target column is either a vocabulary or a spread. Which of
them a column is, is not decided here: the encoder that reads the cells is the one thing that knows
what they hold, so it is the one that answers. What it reads a cell *as* is here, though — one answer
to "does this hold a number", so that a report and the encoder refusing the column cannot part over it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from math import isfinite

import numpy as np

from src.core import ClassDistribution, ValueDistribution


def counted(classes: Mapping[int, str] | None, labels: Iterable[str]) -> ClassDistribution:
    """Count labels, starting from every declared class so the unused ones still show.

    The vocabulary seeds the count, because a class the split never produced is the most useful row on
    the table. A label outside it is still counted rather than dropped, and which of two things that
    means is the encoder's to know: for a declared vocabulary it is a diagnosis, refused when the split
    is validated, and a report should not be what hides it; for one the training split settled it is the
    ordinary picture of an open split — the identities held out are exactly the rows the seed cannot
    hold. Counting either as zero would describe a split that was not read.

    Parameters:
        classes: The declared vocabulary, seeded at zero; ``None`` counts only what is there.
        labels: One label per sample, or per pixel for a mask.
    """
    counts = dict.fromkeys((classes or {}).values(), 0)
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return ClassDistribution(counts=counts)


def as_number(value: object) -> float | None:
    """The number a cell holds, or ``None`` where it holds none — a blank, an ``n/a``, a stray word.

    One reading for the two questions a numeric column is asked: a report drops what is not a number
    and says so in its count, an encoder refuses it, and both have to agree on which cells those are.
    A second ``float()`` beside this one would be free to disagree about ``inf`` or about a cell whose
    type has no conversion at all.

    Infinity counts as no number: it converts, and then it poisons every statistic it is averaged into
    exactly as ``nan`` does, and no annotation cell means it.

    Asked by attempting it, because a cell is whatever the column holds — a ``numpy`` integer is not an
    ``int``, a word that spells a number is a ``str``, and the conversion answers for all of them.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _numbers(values: Iterable[object]) -> Iterator[float]:
    """Every cell that is a number, and only those."""
    for value in values:
        if (number := as_number(value)) is not None:
            yield number


def measured(values: Iterable[object]) -> ValueDistribution | None:
    """The five-number summary of a numeric column, or ``None`` where it holds no number at all.

    A cell that holds no number — a blank, an ``n/a``, a stray word — is dropped rather than carried.
    Which cells those are is ``as_number``'s to say, and the encoder refusing them reads the same answer.
    One of them would otherwise turn every statistic into ``nan``, or end the run from inside a
    display with an unnamed conversion error; the count against the split's row count already says
    how many were missing, which is the honest reading of a column with a gap in it.
    """
    numbers = np.asarray(list(_numbers(values)), dtype=np.float64)
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
