"""What the run is about to train on, said once before the first epoch."""

from __future__ import annotations

import logging

# Runtime imports, not TYPE_CHECKING: ``singledispatch.register`` reads a function's
# annotations when it registers it, so every name in them has to exist by then.
from collections.abc import Iterable, Mapping, Sequence
from functools import singledispatch
from typing import override

import lightning as L
from rich.table import Table

from src.console import HEADER_STYLE, TITLE_STYLE, console
from src.core.entities import ClassDistribution, DatasetStatistics, Distribution, ValueDistribution
from src.core.ports import DataModule
from src.core.reporting import Bars, BarsLogger, BoxPlot, BoxPlotLogger
from src.core.taxonomy import Stage

log = logging.getLogger(__name__)

_SHARE_DECIMALS = 1
"""A class share reads as a percentage with one decimal; more is noise at a glance."""


def _report_table(task: str, measures: str) -> Table:
    """An empty table dressed the way every table this framework prints is dressed.

    Both of the tables below start here, so a change of look is one edit and cannot
    land on one of them only. The task comes first in the title because that is what
    a reader is looking for; what is being measured qualifies it.
    """
    return Table(
        title=f"{task} — {measures}",
        title_justify="left",
        title_style=TITLE_STYLE,
        header_style=HEADER_STYLE,
    )


class DatasetSummary(L.Callback):
    """Report the dataset's size and target distributions once, before anything runs.

    The presentation half of the report: the data module *counts* through
    ``statistics()``, this *shows*. Two audiences get the same numbers by different
    routes — a table in the terminal, where someone is watching the run start, and a
    chart in the tracker, where someone compares two runs a week later.

    Lifecycle only. What a distribution looks like is decided by the two
    ``singledispatch`` points below, which dispatch on the distribution's own class, so
    a new kind of distribution is two registrations and no edits anywhere else.

    Asked of the ``DataModule`` port, which every pipeline answers — one that cannot
    describe its data returns an empty record, and a task whose encoder describes
    nothing is named rather than dropped.

    Parameters:
        title (str): The tracker title the charts land under.
    """

    def __init__(self, title: str = "dataset") -> None:
        super().__init__()
        self._title = title
        self._said = False

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._report(trainer)

    @override
    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Also here, so an evaluation-only run still says what it is evaluating on."""
        self._report(trainer)

    def _report(self, trainer: L.Trainer) -> None:
        if self._said or not trainer.is_global_zero:
            return
        self._said = True
        statistics = _statistics_of(trainer)
        printed = console()
        for task, per_stage in statistics.targets.items():
            if not per_stage:
                log.info("Task '%s' has no distribution to show: its encoder does not describe its column.", task)
                continue
            printed.print(table_for(task, per_stage, statistics.rows))
            draw(_shape_of(per_stage), f"{self._title}/{task}", per_stage, trainer.loggers)


def _statistics_of(trainer: L.Trainer) -> DatasetStatistics:
    """The counts, from whichever pipeline this run is using.

    ``Trainer.datamodule`` is a public runtime attribute the type stubs do not
    declare. ``TrainingData`` adapts our port for Lightning and offers the pipeline
    it wraps as ``source``, so the port is reached through its own accessor rather
    than through the adapter's internals.
    """
    attached = getattr(trainer, "datamodule", None)
    for candidate in (getattr(attached, "source", None), attached):
        if isinstance(candidate, DataModule):
            return candidate.statistics()
    return DatasetStatistics()


def ordered(per_stage: Mapping[Stage, Distribution]) -> list[Stage]:
    """The stages present, in the framework's own order — read off ``Stage``, never re-listed."""
    return [stage for stage in Stage if stage in per_stage]


def _shape_of(per_stage: Mapping[Stage, Distribution]) -> Distribution:
    """Any one of the distributions, to choose an implementation by.

    Every stage of one task is the same kind, because one encoder produced them all
    — so which one is taken cannot matter, and the choice is made here once rather
    than at each call site.
    """
    return per_stage[ordered(per_stage)[0]]


def table_for(task: str, per_stage: Mapping[Stage, Distribution], rows: Mapping[Stage, int]) -> Table:
    """One task's distribution as one table, in whichever shape it has.

    The wrapper exists so no caller has to pass the same distributions twice — once
    to choose the implementation and once as the data.
    """
    return _table(_shape_of(per_stage), task, per_stage, rows)


@singledispatch
def _table(shown: Distribution, task: str, per_stage: Mapping[Stage, Distribution], rows: Mapping[Stage, int]) -> Table:
    """``rows`` is how many samples each stage holds, carried in rather than derived.

    Only a single-label column has as many counts as it has rows. A multilabel one
    counts every label a row carries and a mask counts pixels, so a reader adding up
    a column to learn the stage's size would be wrong by a factor nothing reveals.
    """
    raise TypeError(f"No table for {type(shown).__name__}; known: ClassDistribution, ValueDistribution.")


@_table.register
def _(shown: ClassDistribution, task: str, per_stage: Mapping[Stage, Distribution], rows: Mapping[Stage, int]) -> Table:
    stages = ordered(per_stage)
    counted = [shown for shown in (per_stage[stage] for stage in stages) if isinstance(shown, ClassDistribution)]
    table = _report_table(task, "class balance")
    table.add_column("Class")
    for stage in stages:
        table.add_column(str(stage).capitalize(), justify="right")
    for name in shown.counts:
        cells = [f"{one.counts.get(name, 0)} ({one.shares.get(name, 0.0):.{_SHARE_DECIMALS}%})" for one in counted]
        # A class no split produced is the row worth seeing, so it is marked rather
        # than left as one zero among numbers.
        unseen = all(one.counts.get(name, 0) == 0 for one in counted)
        table.add_row(f"[yellow]{name}[/]" if unseen else name, *cells)
    table.add_section()
    table.add_row("[bold]Total[/]", *(f"[bold]{rows.get(stage, 0)}[/]" for stage in stages))
    return table


_MEASURES: tuple[tuple[str, str], ...] = (
    ("Mean", "mean"),
    ("Std", "deviation"),
    ("Min", "minimum"),
    ("25%", "q25"),
    ("50%", "median"),
    ("75%", "q75"),
    ("Max", "maximum"),
)
"""What the table shows, and the field each column reads — named once, in display order."""


@_table.register
def _(shown: ValueDistribution, task: str, per_stage: Mapping[Stage, Distribution], rows: Mapping[Stage, int]) -> Table:
    stages = ordered(per_stage)
    table = _report_table(task, "value spread")
    table.add_column("Stage")
    table.add_column("Total", justify="right")
    for header, _ in _MEASURES:
        table.add_column(header, justify="right")
    for stage, measured in _measured(per_stage, stages):
        held = _held(rows.get(stage, 0), measured.count)
        table.add_row(str(stage), held, *(_number(getattr(measured, field)) for _, field in _MEASURES))
    return table


def _held(rows: int, counted: int) -> str:
    """How many rows the stage holds — and how many held no number, where any did not.

    One column rather than two. They differ only where a value is missing, so on an
    ordinary column the second repeated the first on every row, and a column that
    repeats its neighbour is noise. Where they do differ that difference is the
    whole point, so it is spelled out rather than left for the reader to subtract.
    """
    missing = rows - counted
    return f"{rows} ({missing} missing)" if missing > 0 else str(rows)


@singledispatch
def draw(shown: Distribution, title: str, per_stage: Mapping[Stage, Distribution], loggers: Iterable[object]) -> None:
    """Send this task's distribution to whichever trackers can draw its shape.

    Each registration builds its own drawable and narrows to its own port, so the
    caller never learns which picture a shape becomes — and a tracker that draws
    neither simply receives nothing.
    """
    raise TypeError(f"No chart for {type(shown).__name__}; known: ClassDistribution, ValueDistribution.")


@draw.register
def _(shown: ClassDistribution, title: str, per_stage: Mapping[Stage, Distribution], loggers: Iterable[object]) -> None:
    stages = ordered(per_stage)
    counted = [one for one in (per_stage[stage] for stage in stages) if isinstance(one, ClassDistribution)]
    labels: Sequence[str] = list(shown.counts)
    bars = Bars(
        series=tuple(str(stage) for stage in stages),
        values=tuple(tuple(float(one.counts.get(name, 0)) for name in labels) for one in counted),
        labels=tuple(labels),
        xaxis="class",
        yaxis="count",
    )
    for drawer in (one for one in loggers if isinstance(one, BarsLogger)):
        drawer.log_bars(title=title, bars=bars, iteration=0)


@draw.register
def _(shown: ValueDistribution, title: str, per_stage: Mapping[Stage, Distribution], loggers: Iterable[object]) -> None:
    stages = ordered(per_stage)
    drawn = BoxPlot(
        series=tuple(str(stage) for stage, _ in _measured(per_stage, stages)),
        boxes=tuple(measured for _, measured in _measured(per_stage, stages)),
        xaxis="stage",
        yaxis="value",
    )
    for drawer in (one for one in loggers if isinstance(one, BoxPlotLogger)):
        drawer.log_box_plot(title=title, box_plot=drawn, iteration=0)


def _measured(
    per_stage: Mapping[Stage, Distribution], stages: Sequence[Stage]
) -> list[tuple[Stage, ValueDistribution]]:
    """The stages whose distribution is a spread, paired with it — narrowing once, by filter."""
    return [(stage, one) for stage in stages if isinstance(one := per_stage[stage], ValueDistribution)]


def _number(value: float) -> str:
    return str(value) if isinstance(value, int) else f"{value:.4g}"
