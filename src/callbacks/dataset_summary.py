"""What the run is about to train on, said once before the first epoch.

Two shapes and two tables, because a target column is either a vocabulary or a spread. Which it is was
decided by the encoder that read the cells; this only shows what came back, and the type checker keeps
the two matches below complete.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import assert_never, override

import lightning as L
from rich.table import Table

from src.callbacks.registry import callback_registry
from src.console import console
from src.core import Bars, ClassDistribution, DatasetStatistics, Distribution, ValueDistribution
from src.tracking import DrawsBars
from src.training import TrainingData

log = logging.getLogger(__name__)

SHARE_DECIMALS = 1
"""A class share reads as a percentage with one decimal; more is noise at a glance."""

MEASURES: tuple[tuple[str, str], ...] = (
    ("Mean", "mean"),
    ("Std", "deviation"),
    ("Min", "minimum"),
    ("25%", "q25"),
    ("50%", "median"),
    ("75%", "q75"),
    ("Max", "maximum"),
)
"""What the spread table shows and the field each column reads — named once, in reading order."""

UNSEEN = "yellow"
"""How a class no split produced is marked. It is the row worth reading, not one zero among numbers."""


@callback_registry.register("dataset_summary")
class DatasetSummary(L.Callback):
    """Report each target's distribution and each split's size, once, before anything runs.

    Declared rather than given to every run, and for one reason: a dense target's class balance is
    counted by reading every mask, which is a pass over the data before the first epoch. The other
    targets cost nothing, but one callback that is sometimes expensive is worse than one a run asks
    for — so this is the line you add when you want to know the data, usually at the start of a project.

    The pipeline counts and this shows: a table in the terminal, where exact numbers are what a reader
    wants, and a chart wherever a backend can draw one, where thirty-seven classes are a glance.

    Parameters:
        title: What the charts are filed under in the tracker.
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
        """Here too, so a run that only evaluates still says what it is evaluating on."""
        self._report(trainer)

    def _report(self, trainer: L.Trainer) -> None:
        if self._said or not trainer.is_global_zero:
            return
        self._said = True
        statistics = _statistics_of(trainer)
        if not statistics:
            log.info("Nothing to summarise: this run's pipeline does not describe what it serves.")
            return
        if not statistics.targets:
            log.info(
                "Nothing to summarise: none of this run's %d rows are read through an encoder that says "
                "what its column holds.",
                sum(statistics.rows.values()),
            )
            return
        printed = console()
        for task, per_split in statistics.targets.items():
            printed.print(table_for(task, per_split, statistics.rows))
            drawn = bars_for(per_split)
            if drawn is not None:
                for backend in (one for one in trainer.loggers if isinstance(one, DrawsBars)):
                    backend.log_bars(f"{self._title}/{task}", drawn, 0)


def table_for(task: str, per_split: Mapping[str, Distribution], rows: Mapping[str, int]) -> Table:
    """One target as one table, in whichever shape its cells turned out to have.

    ``rows`` is how many samples each split holds, carried in rather than derived: only a single-label
    column has as many counts as it has rows, while a multilabel one counts every label a row carries
    and a mask counts pixels — so a reader adding up a column to learn a split's size would be wrong
    by a factor nothing on the table reveals.
    """
    match _any_of(per_split):
        case ClassDistribution():
            return _balance(task, per_split, rows)
        case ValueDistribution():
            return _spread(task, per_split, rows)
        case unknown:
            assert_never(unknown)


def bars_for(per_split: Mapping[str, Distribution]) -> Bars | None:
    """A balance as grouped bars, or nothing for a spread, which the table already says in full."""
    match _any_of(per_split):
        case ClassDistribution():
            balances = _of_kind(per_split, ClassDistribution)
            labels = _labels(balances)
            return Bars(
                series=tuple(balances),
                values=tuple(tuple(float(one.counts.get(name, 0)) for name in labels) for one in balances.values()),
                labels=labels,
                xaxis="class",
                yaxis="count",
            )
        case ValueDistribution():
            return None
        case unknown:
            assert_never(unknown)


def _balance(task: str, per_split: Mapping[str, Distribution], rows: Mapping[str, int]) -> Table:
    balances = _of_kind(per_split, ClassDistribution)
    table = _table(task, "class balance")
    table.add_column("Class")
    for split in balances:
        table.add_column(str(split).capitalize(), justify="right")
    for name in _labels(balances):
        cells = [
            f"{one.counts.get(name, 0)} ({one.shares.get(name, 0.0):.{SHARE_DECIMALS}%})" for one in balances.values()
        ]
        unseen = all(one.counts.get(name, 0) == 0 for one in balances.values())
        table.add_row(f"[{UNSEEN}]{name}[/]" if unseen else name, *cells)
    table.add_section()
    table.add_row("[bold]Total[/]", *(f"[bold]{rows.get(split, 0)}[/]" for split in balances))
    return table


def _spread(task: str, per_split: Mapping[str, Distribution], rows: Mapping[str, int]) -> Table:
    spreads = _of_kind(per_split, ValueDistribution)
    table = _table(task, "value spread")
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    for header, _ in MEASURES:
        table.add_column(header, justify="right")
    for split, one in spreads.items():
        held = _held(rows.get(split, 0), one.count)
        table.add_row(str(split), held, *(_number(getattr(one, field)) for _, field in MEASURES))
    return table


def _labels(balances: Mapping[str, ClassDistribution]) -> tuple[str, ...]:
    """Every class any split counted, in the order they were first seen.

    Across every split rather than from the first: a class only a later one holds would otherwise have
    no row at all, while its rows still counted toward that split's total — the column would read
    11% and 11% and say nothing about the other 78%.
    """
    seen: dict[str, None] = {}
    for one in balances.values():
        seen.update(dict.fromkeys(one.counts))
    return tuple(seen)


def _table(task: str, measures: str) -> Table:
    """An empty table dressed the way both of the ones above are dressed.

    The target comes first in the title because that is what a reader is looking for; what is being
    measured qualifies it.
    """
    return Table(title=f"{task} — {measures}", title_justify="left")


def _held(rows: int, counted: int) -> str:
    """How many rows the split holds, and how many held no number where any did not.

    One column rather than two. They differ only where a value is missing, so on an ordinary column
    the second repeats the first on every row; where they do differ, that difference is the whole
    point, so it is spelled out rather than left to be subtracted.
    """
    missing = rows - counted
    return f"{rows} ({missing} missing)" if missing > 0 else str(rows)


def _of_kind[D: Distribution](per_split: Mapping[str, Distribution], kind: type[D]) -> dict[str, D]:
    """Every split's distribution, narrowed to the one shape they all have.

    One encoder described them all, so they cannot differ. Refused rather than filtered, because a
    filter that never drops anything reads as a guard while the failure it would let through — a split
    quietly missing from the table — is the one nobody would notice.
    """
    narrowed: dict[str, D] = {split: one for split, one in per_split.items() if isinstance(one, kind)}
    if len(narrowed) != len(per_split):
        odd = ", ".join(sorted(per_split.keys() - narrowed.keys()))
        raise TypeError(f"Splits {odd} describe this target as something other than a {kind.__name__}.")
    return narrowed


def _any_of(per_split: Mapping[str, Distribution]) -> Distribution:
    """Any one of them, to choose a shape by.

    Every split of one target is the same kind, because one encoder described them all — so which one
    is taken cannot matter, and the choice is made here once rather than at each call site.
    """
    return next(iter(per_split.values()))


def _statistics_of(trainer: L.Trainer) -> DatasetStatistics:
    """The counts, from the pipeline this run is using.

    ``Trainer.datamodule`` is a public runtime attribute the type stubs do not declare; the adapter
    this framework attaches is what publishes the pipeline's own answer.
    """
    attached = getattr(trainer, "datamodule", None)
    return attached.statistics() if isinstance(attached, TrainingData) else DatasetStatistics()


def _number(value: float) -> str:
    """Four significant figures. Every measure on the spread table is a float; the count is not one."""
    return f"{value:.4g}"
