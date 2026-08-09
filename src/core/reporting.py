"""Routing one computed metric value to wherever its geometry belongs."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from src.core import log_keys
from src.core.entities import Curve, Matrix, PerClass
from src.core.ports import CurveLogger, MatrixLogger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)


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

    Scalars go through ``scalar_log``; a per-class reading becomes its mean plus one
    scalar per class it is about; a *family* of readings is a namespace whose members
    are each routed by their own geometry; drawable artifacts go to **every** backend
    whose port can take them, and to none where none can — a CSV run keeps its scalars
    without an epoch-wise warning about a picture it never asked for.

    **An artifact is drawn only when it is identified.** A ``Curve`` or ``Matrix`` was
    built by whoever knew the metric, so it may be drawn; a raw tuple or a raw 2-D
    tensor arrives without that knowledge — PR and ROC tuples are mirror images of each
    other, and a matrix must not wear class names it may not have — so it warns instead.
    Scalars and vectors need no identification: a vector *is* scalars per class.

    Class names are contributed here, and only into artifacts that left their
    ``labels`` / ``series`` open.

    Parameters:
        key (str): The log key this value was computed under.
        value (Any): Whatever the metric returned; its geometry decides the route.
        scalar_log (Callable[[str, Any], None]): Where a single number goes — the
            training module's own ``self.log``.
        loggers (Iterable[object]): The run's trackers, tested against the artifact ports
            each may or may not implement. Every one that can draw a shape is given it: a
            run configures a second backend precisely so that both receive its results,
            and ``trainer.logger`` — which this took — is only the first of them.
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
            f"{_geometry(value)}); register a presentation for its metric class to draw it.",
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
