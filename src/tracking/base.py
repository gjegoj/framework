"""What a run can show a tracker: one Protocol per shape a backend knows how to draw.

Structural rather than inherited, and one port per shape rather than one fat tracker interface: a
backend implements the shapes it has, and a value is offered only to the backends that can take it.
A tracker that draws nothing still records every number — that path needs no contract of ours,
because Lightning's own ``log`` already is one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core import Matrix


@runtime_checkable
class DrawsMatrix(Protocol):
    """A backend that can draw a two-dimensional reading — a confusion matrix, above all."""

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None: ...


@runtime_checkable
class RecordsSummary(Protocol):
    """A backend with a place for a run's headline numbers, off the axis the rest are drawn on.

    A number reported each epoch is a line; a headline number is one value for the whole run, and a
    backend that keeps the two apart shows the second at a glance — ClearML calls it single values.
    """

    def record_summary(self, name: str, value: float) -> None: ...
