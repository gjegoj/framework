"""What a run reports, the backends that can show it, and the routing between them.

One vocabulary in one place: an artifact arrives completed, so entities carry no behaviour
and ports carry no types of their own. The ports are six role interfaces rather than one
``ArtifactLogger`` so each carries the typed entity its backend draws. Artifacts a metric
returns are not frozen — measured: torchmetrics' ``apply_to_collection`` refuses a frozen
dataclass; ``Bars`` and ``BoxPlot`` never pass through a metric and stay frozen.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from src.core import log_keys
from src.core.entities import ValueDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PerClass:
    """Values a metric produced per class, with which classes they are about.

    Position is *not* the class index: COCO's ``map_per_class`` covers only the classes that
    appeared and says which in ``classes``; a dense reading has ``classes == arange(len(values))``.
    Mutable, as every artifact a metric returns is (see the module docstring).
    """

    values: Tensor
    classes: Tensor

    def __post_init__(self) -> None:
        if len(self.values) != len(self.classes):
            raise ValueError(
                f"PerClass needs one class per value, got {len(self.values)} values and {len(self.classes)} classes."
            )


@dataclass(slots=True)
class Curve:
    """A curve metric's plotted lines, already oriented for drawing.

    PR and ROC share one geometry with opposite axes, so orientation is stated by the metric
    that knew it. One entry per class; a binary metric carries the positive class's line.
    ``series is None`` means the lines live in the task's class space and the router fills
    the names. Mutable, as every artifact a metric returns is.
    """

    x: tuple[Tensor, ...]
    y: tuple[Tensor, ...]
    xaxis: str
    yaxis: str
    positive_only: bool = False
    series: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"A curve needs x and y per line, got {len(self.x)} x and {len(self.y)} y.")
        if self.series is not None and len(self.series) != len(self.x):
            raise ValueError(f"A curve with {len(self.x)} lines cannot carry {len(self.series)} series names.")


@dataclass(slots=True)
class Matrix:
    """A drawable 2-D artifact, axes named by the metric that knew them.

    ``labels is None`` means the index space is the task's classes and the router fills the
    names. Mutable, as every artifact a metric returns is.
    """

    value: Tensor
    xaxis: str
    yaxis: str
    labels: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Bars:
    """Named quantities drawn as grouped bars — a class balance across stages.

    One series per group and one value per label within it, so a class missing from one
    split is a gap rather than a number to hunt for.
    """

    series: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]
    xaxis: str
    yaxis: str


@dataclass(frozen=True, slots=True)
class BoxPlot:
    """Five-number summaries drawn as boxes — one per series, on shared axes.

    Carries the ``ValueDistribution``s themselves, not a copy of their numbers. Whiskers are
    the observed minimum and maximum, not Tukey's fences: outliers would need the raw values
    held in memory for a picture drawn once.
    """

    series: tuple[str, ...]
    boxes: tuple[ValueDistribution, ...]
    xaxis: str
    yaxis: str


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


def report_metric(
    key: str,
    value: Any,
    *,
    scalar_log: Callable[[str, Any], None],
    loggers: Iterable[object],
    step: int,
    class_names: list[str] | None,
) -> None:
    """Deliver one computed metric value to wherever its geometry belongs.

    Scalars go through ``scalar_log``; a per-class reading becomes its mean plus one scalar
    per class; a family of readings is a namespace routed member by member; a drawable
    artifact goes to every backend whose port can take it. An artifact is drawn only when it
    is identified — a raw tuple or 2-D tensor arrives without that knowledge and warns
    instead. Class names are filled in only where an artifact left ``labels`` / ``series`` open.

    Parameters:
        key (str): The log key this value was computed under.
        value (Any): Whatever the metric returned; its geometry decides the route.
        scalar_log (Callable[[str, Any], None]): Where a single number goes.
        loggers (Iterable[object]): Every run tracker; each that can draw a shape is given it.
        step (int): Iteration the value belongs to.
        class_names (list[str] | None): The task's class space, where it declared one.
    """
    if isinstance(value, Mapping):
        # `{stage}/{task}/{metric}/{reading}` already draws a family on one graph, so
        # the members need routing rather than a grammar of their own.
        for reading, inner in value.items():
            report_metric(
                log_keys.join(key, str(reading)),
                inner,
                scalar_log=scalar_log,
                loggers=loggers,
                step=step,
                class_names=class_names,
            )
    elif isinstance(value, Curve):
        # Completed once, outside the loop: naming a curve's series is about the task,
        # not about which backend happens to draw it.
        drawn_curve = _completed_curve(key, value, class_names)
        for drawer in (one for one in loggers if isinstance(one, CurveLogger)):
            drawer.log_curve(title=key, curve=drawn_curve, iteration=step)
    elif isinstance(value, Matrix):
        drawn_matrix = _completed_matrix(value, class_names)
        for plotter in (one for one in loggers if isinstance(one, MatrixLogger)):
            plotter.log_matrix(title=key, matrix=drawn_matrix, iteration=step)
    elif isinstance(value, tuple) or (isinstance(value, Tensor) and value.ndim >= 2):
        warnings.warn(
            f"Metric '{key}' returned an unidentified artifact ({type(value).__name__}, "
            f"{_geometry(value)}); a metric draws by returning a Curve, a Matrix or a "
            "PerClass from compute — see WrappedMetric.",
            stacklevel=2,
        )
    elif isinstance(value, PerClass):
        _report_per_class(key, value, scalar_log=scalar_log, class_names=class_names)
    elif not isinstance(value, Tensor) or value.ndim == 0:
        scalar_log(key, value)
    else:
        # A dense vector is the case where position *is* the class, so it becomes one
        # and reuses the naming path rather than getting a second that has to agree.
        _report_per_class(
            key,
            PerClass(value, torch.arange(value.shape[0])),
            scalar_log=scalar_log,
            class_names=class_names,
        )


def _report_per_class(
    key: str,
    reading: PerClass,
    *,
    scalar_log: Callable[[str, Any], None],
    class_names: list[str] | None,
) -> None:
    """One scalar per class it is about, plus the mean they are read against.

    Named through ``classes`` rather than by position: a sparse reading covers only the
    classes that appeared, and position would put one class's number under another's name.
    """
    scalar_log(log_keys.join(key, log_keys.MEAN), reading.values.float().mean())
    for index, value in zip(reading.classes.tolist(), reading.values, strict=True):
        scalar_log(log_keys.join(key, _class_label(key, int(index), class_names)), value.float())


def _class_label(key: str, index: int, class_names: list[str] | None) -> str:
    """The class's name where the task declared one, and its index where it did not."""
    if class_names is not None and 0 <= index < len(class_names):
        return class_names[index]
    if class_names is not None:
        warnings.warn(
            f"Metric '{key}' reports class {index}, which the task's {len(class_names)} declared "
            f"class name(s) do not cover; it is logged by index.",
            stacklevel=3,
        )
    return f"class{index}"


def _completed_curve(key: str, curve: Curve, class_names: list[str] | None) -> Curve:
    """Series names from the task's class space, only where the curve left them open."""
    if curve.series is not None:
        return curve
    if curve.positive_only:
        positive = class_names[1] if class_names is not None and len(class_names) >= 2 else "positive"
        return replace(curve, series=(positive,))
    return replace(curve, series=tuple(_aligned_names(key, len(curve.x), class_names)))


def _completed_matrix(matrix: Matrix, class_names: list[str] | None) -> Matrix:
    """Axis labels from the task's class space, only where the matrix left them open."""
    if matrix.labels is not None or class_names is None:
        return matrix
    return replace(matrix, labels=tuple(class_names))


def _aligned_names(key: str, size: int, class_names: list[str] | None) -> list[str]:
    """Per-class names, warning once when the declared names cannot align."""
    if class_names is not None and len(class_names) != size:
        warnings.warn(
            f"Metric '{key}' has {size} values but {len(class_names)} class names; falling back to indexed labels.",
            stacklevel=3,
        )
        class_names = None
    if class_names is None:
        return [f"class{index}" for index in range(size)]
    return class_names


def _geometry(value: Any) -> str:
    if isinstance(value, Tensor):
        return f"shape {tuple(value.shape)}"
    return f"{len(value)} elements"


__all__ = ["report_metric"]
