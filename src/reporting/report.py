"""Thin dispatcher: print distribution tables and log plots for every task.

``report_dataset_statistics`` is the single entry point consumed by
``DatasetStatsCallback``. It delegates all rendering to the
``distribution_renderers`` registry — no ``isinstance`` branching on
distribution type; adding a new distribution type only requires a new
``DistributionRenderer`` registered in ``renderers.py``.
"""

from __future__ import annotations

from rich.console import Console

from src.data.statistics import DatasetStatistics
from src.reporting.renderers import _ordered_stages, distribution_renderers


def report_dataset_statistics(statistics: DatasetStatistics, logger: object) -> None:
    """Print per-task distribution tables and log plots to ``logger``.

    For each task in *statistics*, the correct ``DistributionRenderer`` is
    looked up by the distribution type's name — no branching on distribution
    type here. Categorical tasks log grouped histograms (behaviour unchanged);
    continuous tasks log an interactive box plot instead of per-stage histograms.

    Parameters:
        statistics (DatasetStatistics): ``{task: {stage: distribution}}`` from
            ``DataModule.statistics()``.
        logger (object): Active logger; each renderer logs plots only when it implements
            that renderer's artifact port (otherwise just the terminal tables are shown).
    """
    if not statistics:
        return
    console = Console()
    console.print("\n[bold blue]Dataset distribution[/]")
    for task_name, per_stage in statistics.items():
        stages = _ordered_stages(per_stage)
        distribution_type_name = type(per_stage[stages[0]]).__name__
        renderer = distribution_renderers.create(distribution_type_name)
        console.print(renderer.table(task_name, per_stage))
        # The renderer narrows ``logger`` to its own artifact port and no-ops if unsupported.
        renderer.log(task_name, per_stage, logger)
