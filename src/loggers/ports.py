"""What a run can show a tracker: one Protocol per shape a backend can draw.

Six role Protocols rather than one ``ArtifactLogger``, so each carries the typed entity its
backend draws and a backend implements only the shapes it has; ``report_metric`` routes a
metric's value by its shape to every logger whose port takes it.

Contracts only, so a capability that publishes one shape imports this and not the routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.data.statistics import Bars, BoxPlot
    from src.metrics.entities import Curve, Matrix


@runtime_checkable
class CurveLogger(Protocol):
    """A backend that can draw an x-y curve artifact (PR, ROC) — all lines at once."""

    def log_curve(self, title: str, curve: Curve, iteration: int) -> None: ...


@runtime_checkable
class MatrixLogger(Protocol):
    """A backend that can draw a 2-D matrix artifact.

    Structural: a backend qualifies by having the method, and one without it keeps its scalars.
    """

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None: ...


@runtime_checkable
class BarsLogger(Protocol):
    """A backend that can draw grouped bars — a dataset's class balance across stages."""

    def log_bars(self, title: str, bars: Bars, iteration: int) -> None: ...


@runtime_checkable
class BoxPlotLogger(Protocol):
    """A backend that can draw boxes — a numeric target's spread, one box per stage."""

    def log_box_plot(self, title: str, box_plot: BoxPlot, iteration: int) -> None: ...


@runtime_checkable
class SingleValueLogger(Protocol):
    """A backend with an end-of-run summary table for headline scalars.

    Distinct from per-step scalars: a value here has no iteration axis —
    ClearML collects them in its "Single Values" table.
    """

    def log_single_value(self, name: str, value: float) -> None: ...


@runtime_checkable
class HtmlLogger(Protocol):
    """A backend that can carry a self-contained HTML page as a run artifact.

    A tracker that can show a page gets one; one that cannot is told so once instead of
    failing a run over a picture.
    """

    def log_html(self, title: str, html: str, iteration: int) -> None: ...


@runtime_checkable
class TagsRuns(Protocol):
    """A backend that can tag a run after it was opened.

    What the composition root knows and config cannot say — the model's architecture — reaches the
    tracker through this, after construction, so a logger built by ``_target_`` that has
    no notion of tags is left alone rather than handed an argument it would refuse.
    """

    def tag_run(self, architecture: str | None) -> None: ...
