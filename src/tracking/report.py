"""How a computed value reaches whatever can show it: one router, rather than a branch per caller.

A metric's geometry decides the route — a number is logged, a reading per class becomes a graph of
lines, an image goes to the backends that draw images. The training module hands values over
without knowing which is which, and a metric announces what it produced by the type it returns.

The set of shapes is closed, deliberately: a value that means a *new* kind of image — a curve, a
histogram — arrives with a value type in ``core``, a port beside ``DrawsMatrix`` and a branch here, and
that is four small edits rather than one. Until a second such shape exists there is nothing to
generalise over, and the shapes that stay unroutable are named out loud instead of dropped.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from torch import Tensor

from src.core import Matrix, class_name
from src.tracking.base import DrawsMatrix
from src.tracking.keys import MEAN, SEGMENT, MetricKey

type ScalarLog = Callable[[str, Any], None]
"""Where a single number goes; ``LightningModule.log`` is the one a run passes in."""


def report(
    key: MetricKey,
    value: object,
    *,
    scalar_log: ScalarLog,
    trackers: Sequence[object] = (),
    step: int,
    classes: Mapping[int, str] | None = None,
) -> None:
    """Deliver one computed value to wherever its shape belongs.

    Parameters:
        key: What this value was computed under; leaves are named below it, so a graph holds a family.
        value: Whatever the metric returned — its geometry is the routing decision.
        scalar_log: Where a single number goes.
        trackers: The run's trackers; each is offered a shape only if it can draw it.
        step: The iteration the value belongs to, which for an epoch-end reading is the epoch.
        classes: The vocabulary this task declared, where it declared one, to name per-class readings.
    """
    match value:
        case Matrix():
            _draw(key, _named_rows(value, classes), trackers=trackers, step=step)
        case Tensor() if value.ndim == 1:
            _per_class(key, value, scalar_log=scalar_log, classes=classes)
        case Tensor() if value.ndim > 1:
            _unshowable(key, f"a {value.ndim}-dimensional tensor")
        case tuple():
            _unshowable(key, f"a tuple of {len(value)}")
        case _:
            scalar_log(str(key), value)


def _per_class(key: MetricKey, values: Tensor, *, scalar_log: ScalarLog, classes: Mapping[int, str] | None) -> None:
    """One number per class, and the mean they are read against, as leaves of one family.

    Named through the vocabulary where there is one *and* it fits: a per-output reading of a numeric
    target has no classes at all, and a samplewise one has as many values as the batch had rows —
    naming either through a vocabulary would put one class's name on another thing's number.
    """
    named = classes if classes is not None and len(classes) == len(values) else None
    scalar_log(str(_leaf(key, MEAN)), values.float().mean())
    for index, value in enumerate(values):
        scalar_log(str(_leaf(key, class_name(named, index))), value.float())


def _named_rows(matrix: Matrix, classes: Mapping[int, str] | None) -> Matrix:
    """A reading's rows named by the task's vocabulary: the metric counted, the task knows the names."""
    if matrix.labels is not None or classes is None:
        return matrix
    return replace(matrix, labels=tuple(classes[index] for index in sorted(classes)))


def _draw(key: MetricKey, matrix: Matrix, *, trackers: Sequence[object], step: int) -> None:
    """Every tracker that can draw a matrix is given it; one that cannot keeps its numbers."""
    drawers = [one for one in trackers if isinstance(one, DrawsMatrix)]
    for drawer in drawers:
        drawer.log_matrix(str(key), matrix, step)
    if trackers and not drawers:
        # Named without its stage, because the answer is the same in every one of them: said once per
        # run rather than once per stage. Silent where a run declared no tracker at all — as asked.
        warnings.warn(
            f"{key.series} is an image, and nothing this run records to can draw one: it is the one reading "
            "that goes unkept. `tracker: clearml` draws it, `tracker: csv` holds numbers only.",
            stacklevel=3,
        )


def _unshowable(key: MetricKey, geometry: str) -> None:
    warnings.warn(
        f"{key} returned {geometry}, which is neither a number nor a reading anything was told how to "
        "draw, so it is not reported. A metric whose value means an image returns one — see "
        "ConfusionMatrix, which returns a Matrix.",
        stacklevel=3,
    )


def _leaf(key: MetricKey, name: str) -> MetricKey:
    return replace(key, name=f"{key.name}{SEGMENT}{name}")
