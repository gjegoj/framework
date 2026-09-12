"""What a run can show a tracker: one Protocol per shape a backend knows how to draw.

Structural rather than inherited, and one port per shape rather than one fat tracker interface: a
backend implements the shapes it has, and a value is offered only to the backends that can take it.
A tracker that draws nothing still records every number — that path needs no contract of ours,
because Lightning's own ``log`` already is one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core import Bars, Matrix


@runtime_checkable
class DrawsMatrix(Protocol):
    """A backend that can draw a two-dimensional reading — a confusion matrix, above all."""

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None: ...


@runtime_checkable
class DrawsBars(Protocol):
    """A backend that can draw grouped bars — a target's class balance across the run's splits.

    An image and not a table, because a balance is read by comparing bar heights and thirty-seven
    breeds is a scroll rather than a glance. The table is printed anyway, in the terminal, where the
    exact numbers are what a reader wants.
    """

    def log_bars(self, title: str, bars: Bars, iteration: int) -> None: ...


@runtime_checkable
class ShowsPage(Protocol):
    """A backend that can carry a self-contained HTML page as part of a run.

    A grid of samples is a page and not an image: it carries the controls that narrow it, and a
    backend that could only keep an image would be keeping a screenshot of one.
    """

    def log_html(self, title: str, html: str, iteration: int) -> None: ...


@runtime_checkable
class RecordsSummary(Protocol):
    """A backend with a place for a run's headline numbers, off the axis the rest are drawn on.

    A number reported each epoch is a line; a headline number is one value for the whole run, and a
    backend that keeps the two apart shows the second at a glance — ClearML calls it single values.
    """

    def record_summary(self, name: str, value: float) -> None: ...
