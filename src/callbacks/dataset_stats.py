"""Dataset-distribution report: print tables and log histograms before the first stage.

Clean split from the data layer: the ``DataModule`` *computes* the distributions
(``statistics()``); this callback *presents* them — a compact rich table per task in the
terminal and a grouped-bar histogram per task to the experiment logger. ``report_dataset_statistics``
is the pure, testable renderer; ``DatasetStatsCallback`` is the thin lifecycle glue that
fires it once, before training (or before eval in an eval-only run).
"""

from __future__ import annotations

from typing import cast

import lightning as L
from rich.console import Console
from rich.table import Table

from src.core.enums import Stage
from src.core.ports import PlotLogger
from src.data.statistics import (
    CategoricalDistribution,
    ContinuousDistribution,
    DatasetStatistics,
    Distribution,
)
from src.training.modules import LitDataModule


class DatasetStatsCallback(L.Callback):
    """Report dataset distributions once, before the first stage runs.

    Reads the distributions from the data module (``statistics()``) and renders them to the
    terminal plus the logger's histograms. A no-op on non-zero ranks, when the data module
    cannot report statistics, or — for the histograms — without a plot-capable logger.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reported = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._report_once(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._report_once(trainer)

    def _report_once(self, trainer: L.Trainer) -> None:
        # ``Trainer.datamodule`` is a public runtime attribute the type stubs do not expose.
        datamodule = getattr(trainer, "datamodule", None)
        if self._reported or not trainer.is_global_zero or not isinstance(datamodule, LitDataModule):
            return
        self._reported = True
        report_dataset_statistics(datamodule.statistics(), trainer.logger)


# --------------------------------------------------------------------- rendering

# Stage order for table columns and histogram series.
_STAGE_ORDER: tuple[Stage, ...] = (Stage.TRAIN, Stage.VAL, Stage.TEST)

# Continuous table rows: (display label, ContinuousDistribution attribute).
_CONTINUOUS_ROWS: tuple[tuple[str, str], ...] = (
    ("count", "count"),
    ("mean", "mean"),
    ("std", "std"),
    ("min", "minimum"),
    ("q25", "q25"),
    ("median", "median"),
    ("q75", "q75"),
    ("max", "maximum"),
)


def report_dataset_statistics(statistics: DatasetStatistics, logger: object) -> None:
    """Print per-task distribution tables and log histograms to ``logger``.

    Parameters:
        statistics (DatasetStatistics): ``{task: {stage: distribution}}`` from
            ``DataModule.statistics()``.
        logger (object): Active logger; histograms are logged only when it is a
            ``PlotLogger`` (otherwise just the terminal tables are shown).
    """
    if not statistics:
        return
    console = Console()
    console.print("\n[bold blue]Dataset distribution[/]")
    for task_name, per_stage in statistics.items():
        _print_task(console, task_name, per_stage)
        if isinstance(logger, PlotLogger):
            _log_task(logger, task_name, per_stage)


def _ordered_stages(per_stage: dict[Stage, Distribution]) -> list[Stage]:
    return [stage for stage in _STAGE_ORDER if stage in per_stage]


def _print_task(console: Console, task_name: str, per_stage: dict[Stage, Distribution]) -> None:
    stages = _ordered_stages(per_stage)
    # A task's encoder yields one distribution type across all stages — check once, then cast.
    if isinstance(per_stage[stages[0]], CategoricalDistribution):
        console.print(_categorical_table(task_name, cast("dict[Stage, CategoricalDistribution]", per_stage), stages))
    else:
        console.print(_continuous_table(task_name, cast("dict[Stage, ContinuousDistribution]", per_stage), stages))


def _new_table(title: str, first_column: str, stages: list[Stage]) -> Table:
    """A titled table with a label column and one right-aligned column per stage."""
    table = Table(title=title, title_justify="left", header_style="bold magenta")  # match the progress bar
    table.add_column(first_column)
    for stage in stages:
        table.add_column(stage.value.capitalize(), justify="right")
    return table


def _categorical_table(task_name: str, per_stage: dict[Stage, CategoricalDistribution], stages: list[Stage]) -> Table:
    table = _new_table(f"{task_name}  (class distribution)", "Class", stages)
    class_names = list(per_stage[stages[0]].counts)  # class-index order, shared across stages
    totals = {stage: per_stage[stage].total for stage in stages}  # once per stage, not per cell
    for class_name in class_names:
        cells = [class_name]
        for stage in stages:
            count = per_stage[stage].counts[class_name]
            percent = (100 * count / totals[stage]) if totals[stage] else 0.0
            cells.append(f"{count} ({percent:.1f}%)")
        table.add_row(*cells)
    table.add_row("Total", *(str(totals[stage]) for stage in stages), style="bold")
    return table


def _continuous_table(task_name: str, per_stage: dict[Stage, ContinuousDistribution], stages: list[Stage]) -> Table:
    table = _new_table(f"{task_name}  (numeric)", "Statistic", stages)
    for label, attribute in _CONTINUOUS_ROWS:
        cells = [label]
        for stage in stages:
            value = getattr(per_stage[stage], attribute)
            cells.append(str(int(value)) if attribute == "count" else f"{value:.3f}")
        table.add_row(*cells)
    return table


def _log_task(logger: PlotLogger, task_name: str, per_stage: dict[Stage, Distribution]) -> None:
    for stage in _ordered_stages(per_stage):
        distribution = per_stage[stage]
        values, labels = _histogram_bars(distribution)
        if isinstance(distribution, CategoricalDistribution):
            # Stages share the class order → one grouped plot, a series (bar group) per stage.
            logger.log_histogram(title=f"dataset/{task_name}", series=stage.value, values=values, labels=labels)
        else:
            # Continuous bins differ per stage; grouping would misalign them → one plot per stage.
            logger.log_histogram(
                title=f"dataset/{task_name}/{stage.value}", series="distribution", values=values, labels=labels
            )


def _histogram_bars(distribution: Distribution) -> tuple[list[float], list[str]]:
    """Bar heights and x-labels: per class for categorical, per bin for continuous."""
    if isinstance(distribution, CategoricalDistribution):
        return [float(count) for count in distribution.counts.values()], list(distribution.counts)
    histogram = distribution.histogram
    return [float(count) for count in histogram.counts], [f"{center:.2f}" for center in histogram.centers]
