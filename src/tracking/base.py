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
