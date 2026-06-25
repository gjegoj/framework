"""Distribution renderers: terminal table + logger output for each distribution type.

Two registries-plus-ABC pattern — mirrors ``src/visualization/renderer.py``:

- ``DistributionRenderer`` (ABC): one ``table`` method (rich terminal) and one
  ``log`` method (experiment logger).
- ``distribution_renderers`` registry: keyed by distribution type ``__name__``;
  ``CategoricalDistributionRenderer`` and ``ContinuousDistributionRenderer`` are
  registered at import time.

Adding a new distribution type = one new subclass + one ``@distribution_renderers.register``.
No ``isinstance`` branching on distribution type anywhere in the dispatch path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from rich.table import Table

from src.core.enums import Stage
from src.core.plotting import BoxPlot, BoxStats
from src.core.ports import HistogramLogger, PlotLogger
from src.core.registry import Registry
from src.data.statistics import (
    CategoricalDistribution,
    ContinuousDistribution,
    Distribution,
)
from src.reporting.tables import categorical_table, continuous_table

# Stage order for table columns and histogram series.
_STAGE_ORDER: tuple[Stage, ...] = (Stage.TRAIN, Stage.VAL, Stage.TEST)


def _ordered_stages(per_stage: dict[Stage, Distribution]) -> list[Stage]:
    """Return stages present in *per_stage* in canonical display order."""
    return [stage for stage in _STAGE_ORDER if stage in per_stage]


class DistributionRenderer(ABC):
    """Renders one task's per-stage distribution to a terminal table and an experiment logger.

    Implementations are registered in ``distribution_renderers`` keyed by the
    distribution type's ``__name__`` so the dispatcher can look them up without
    any ``isinstance`` branching on distribution type.
    """

    @abstractmethod
    def table(self, task_name: str, per_stage: dict[Stage, Distribution]) -> Table:
        """Build the rich terminal table for this task.

        Parameters:
            task_name (str): Task name used as the table title prefix.
            per_stage (dict[Stage, Distribution]): Distribution per stage.

        Returns:
            Table: A rich ``Table`` ready to be printed to the console.
        """

    @abstractmethod
    def log(self, task_name: str, per_stage: dict[Stage, Distribution], logger: object) -> None:
        """Log the distribution to the experiment logger.

        The renderer narrows ``logger`` to the artifact port it needs and no-ops when
        the active logger does not implement it (so the caller never branches on type).

        Parameters:
            task_name (str): Task name used in the plot title.
            per_stage (dict[Stage, Distribution]): Distribution per stage.
            logger (object): Active logger; used only if it implements this
                renderer's artifact port (``HistogramLogger`` / ``PlotLogger``).
        """


distribution_renderers: Registry[DistributionRenderer] = Registry("distribution_renderer")


@distribution_renderers.register(CategoricalDistribution.__name__)
class CategoricalDistributionRenderer(DistributionRenderer):
    """Renderer for ``CategoricalDistribution``: class-count table + grouped log_histogram."""

    def table(self, task_name: str, per_stage: dict[Stage, Distribution]) -> Table:
        """Build the class-distribution table.

        Parameters:
            task_name (str): Task name for the table title.
            per_stage (dict[Stage, Distribution]): Per-stage categorical distributions.

        Returns:
            Table: Rich table with one row per class and a totals row.
        """
        stages = _ordered_stages(per_stage)
        return categorical_table(task_name, cast("dict[Stage, CategoricalDistribution]", per_stage), stages)

    def log(self, task_name: str, per_stage: dict[Stage, Distribution], logger: object) -> None:
        """Log one grouped histogram (a series per stage) to the experiment logger.

        No-ops unless ``logger`` is a ``HistogramLogger``. Stages share the class order
        so grouping is coherent — all bars for each class align across stages.

        Parameters:
            task_name (str): Task name used as the plot title (``"dataset/{task_name}"``).
            per_stage (dict[Stage, Distribution]): Per-stage categorical distributions.
            logger (object): Receives one ``log_histogram`` call per stage when it is a ``HistogramLogger``.
        """
        if not isinstance(logger, HistogramLogger):
            return
        for stage in _ordered_stages(per_stage):
            distribution = cast(CategoricalDistribution, per_stage[stage])
            values = [float(count) for count in distribution.counts.values()]
            labels = list(distribution.counts)
            logger.log_histogram(title=f"dataset/{task_name}", series=stage.value, values=values, labels=labels)


@distribution_renderers.register(ContinuousDistribution.__name__)
class ContinuousDistributionRenderer(DistributionRenderer):
    """Renderer for ``ContinuousDistribution``: numeric-summary table + log_plot box plot."""

    def table(self, task_name: str, per_stage: dict[Stage, Distribution]) -> Table:
        """Build the numeric-summary table.

        Parameters:
            task_name (str): Task name for the table title.
            per_stage (dict[Stage, Distribution]): Per-stage continuous distributions.

        Returns:
            Table: Rich table with one row per summary statistic.
        """
        stages = _ordered_stages(per_stage)
        return continuous_table(task_name, cast("dict[Stage, ContinuousDistribution]", per_stage), stages)

    def log(self, task_name: str, per_stage: dict[Stage, Distribution], logger: object) -> None:
        """Log a single box plot covering all stages to the experiment logger.

        No-ops unless ``logger`` is a ``PlotLogger``. One ``BoxPlot`` with one
        ``BoxStats`` per stage gives a richer interactive view (quartile fences, mean
        marker) than binned bars.

        Parameters:
            task_name (str): Task name used as the plot title (``"dataset/{task_name}"``).
            per_stage (dict[Stage, Distribution]): Per-stage continuous distributions.
            logger (object): Receives one ``log_plot`` call when it is a ``PlotLogger``.
        """
        if not isinstance(logger, PlotLogger):
            return
        stages = _ordered_stages(per_stage)
        boxes: list[BoxStats] = []
        for stage in stages:
            distribution = cast(ContinuousDistribution, per_stage[stage])
            boxes.append(
                BoxStats(
                    minimum=distribution.minimum,
                    q25=distribution.q25,
                    median=distribution.median,
                    q75=distribution.q75,
                    maximum=distribution.maximum,
                    mean=distribution.mean,
                )
            )
        box_plot = BoxPlot(
            title=f"dataset/{task_name}",
            categories=[stage.value for stage in stages],
            boxes=boxes,
            y_label=task_name,
        )
        logger.log_plot(box_plot)


__all__ = [
    "DistributionRenderer",
    "distribution_renderers",
    "CategoricalDistributionRenderer",
    "ContinuousDistributionRenderer",
]
