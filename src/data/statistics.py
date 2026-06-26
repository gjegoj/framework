"""Dataset distribution entities — the result of summarizing target columns.

Domain value objects (frozen dataclasses, not Pydantic). Two shapes cover every
target: a **categorical** distribution (count per class — classification,
multilabel, and, later, segmentation pixel counts) and a **continuous**
distribution (numeric summary + histogram — regression). The reporter renders
whichever shape it is handed, so a new producer (e.g. ``MaskEncoder.summarize``)
needs no reporter change.

``SupportsSummary`` is an optional-capability Protocol: encoders that can produce
a distribution declare it by implementing ``summarize``; the base ``TargetEncoder``
ABC does not.  Use ``isinstance(encoder, SupportsSummary)`` to guard before calling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.core.enums import Stage


@dataclass(frozen=True, slots=True)
class CategoricalDistribution:
    """Count of samples per class label, in class-index order.

    Parameters:
        counts (dict[str, int]): Class label → count (includes zero-count classes).
    """

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def relative(self) -> dict[str, float]:
        """Class label → fraction of the total (``0.0`` when the total is zero)."""
        total = self.total
        return {label: (count / total if total else 0.0) for label, count in self.counts.items()}


@dataclass(frozen=True, slots=True)
class ContinuousDistribution:
    """Summary statistics of a numeric target.

    Parameters:
        count (int): Number of non-null values.
        mean, std, minimum, q25, median, q75, maximum (float): Summary statistics.
    """

    count: int
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float


type Distribution = CategoricalDistribution | ContinuousDistribution


@runtime_checkable
class SupportsSummary(Protocol):
    """Optional-capability Protocol for target encoders that can describe their column's distribution.

    Encoders implementing this Protocol return a ``Distribution`` from ``summarize``; the
    base ``TargetEncoder`` ABC does not declare it.  Callers must gate on
    ``isinstance(encoder, SupportsSummary)`` before calling ``summarize``.

    Parameters:
        values (Iterable[Any]): Raw column values for one stage, as stored in the CSV / source frame.

    Returns:
        Distribution | None: The distribution, or ``None`` when the encoder cannot summarize
            (e.g. empty data, deferred implementation).
    """

    def summarize(self, values: Iterable[Any]) -> Distribution | None: ...


# Per-task distributions across stages: ``{task_name: {stage: distribution}}``.
type DatasetStatistics = dict[str, dict[Stage, Distribution]]
