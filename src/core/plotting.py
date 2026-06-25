"""Backend-agnostic plot data transfer objects.

Pure data layer — no Plotly import. Concrete figure-building lives in
``src/loggers/plotly.py``, which is the only module that may import Plotly.

The ``type Plot`` alias mirrors ``type Distribution`` in ``src/data/statistics.py``
and grows by union as new plot types are added.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoxStats:
    """Five-number summary plus mean for one box.

    Field names mirror ``ContinuousDistribution`` so callers can map one-to-one
    without renaming.

    Parameters:
        minimum (float): Minimum observed value (lower fence of the box).
        q25 (float): First quartile (25th percentile).
        median (float): Median (50th percentile).
        q75 (float): Third quartile (75th percentile).
        maximum (float): Maximum observed value (upper fence of the box).
        mean (float): Arithmetic mean (rendered as a marker overlay).
    """

    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    mean: float


@dataclass(frozen=True, slots=True)
class BoxPlot:
    """A grouped box plot: one box per category.

    Each element of ``boxes`` corresponds to the same-index element of
    ``categories`` (e.g. one box per dataset stage: train / val / test).

    Parameters:
        title (str): Display title shown at the top of the figure.
        categories (list[str]): X-axis category label per box.
        boxes (list[BoxStats]): Per-category five-number summary + mean.
        y_label (str): Y-axis label (default ``"value"``).
    """

    title: str
    categories: list[str]
    boxes: list[BoxStats]
    y_label: str = "value"


# Union alias — grows as new plot types are added, mirroring `type Distribution`.
type Plot = BoxPlot
