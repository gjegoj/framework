"""Mean average precision over the framework's own ``Instances``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from src.core.entities import Instances
from src.core.reporting import PerClass
from src.metrics.adapter import WrappedMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.entities import TaskOutput

DEFAULT_READINGS: Final = ("map", "map_50", "map_75")
"""What a detection run is read by: the COCO average, and the two thresholds quoted beside it.

Three of fifteen. The rest — the per-scale breakdowns, the recall family, the per-class
readings — are reached by naming them, because a graph carrying fifteen series from the
first run is not a report.
"""

PER_CLASS_READINGS: Final = frozenset({"map_per_class", "mar_100_per_class"})
"""The readings torchmetrics produces only when told to, and the reason it is told."""

_NOT_APPLICABLE: Final = -1.0
"""What torchmetrics returns for a reading its data could not support.

Measured on ``map_large`` in a split holding only small boxes. Mean average precision
lives in ``[0, 1]``, so this is never a measurement — logged as one it would drag a chart
and let a checkpoint monitor rank it.
"""

_INDEX = "classes"
"""The key saying which class each per-class reading belongs to — not a reading itself."""


class MeanAveragePrecisionOverInstances(WrappedMetric):
    """Mean average precision, taking ``Instances`` on both sides and publishing a family.

    A ``WrappedMetric`` rather than a second ``MetricSet``: the input shape is knowledge
    of the metric, so the metric set stays a pass-through and every other metric is
    unaffected. Its ``compute`` returns ``PerClass`` for the per-class readings, which is
    the whole of how a metric says what its value means.

    torchmetrics computes all fifteen of its readings in one pass, so asking for three
    costs exactly what asking for one would. A second entry earns its place only when its
    *arguments* differ.

    Parameters:
        readings (Sequence[str]): Which of the computed readings to publish. Each becomes
            a key of its own under the entry's label, so they share one graph.
        **kwargs (Any): Forwarded verbatim to ``MeanAveragePrecision`` — ``iou_thresholds``,
            ``max_detection_thresholds``, ``rec_thresholds`` and the rest stay reachable.
    """

    higher_is_better = True

    def __init__(self, readings: Sequence[str] = DEFAULT_READINGS, **kwargs: Any) -> None:
        from torchmetrics.detection import MeanAveragePrecision

        super().__init__(
            MeanAveragePrecision(
                box_format="xyxy",
                # Named rather than left to the default: torchmetrics reaches for
                # `pycocotools` and raises for it even where `faster-coco-eval` is the
                # installed backend.
                backend="faster_coco_eval",
                # Derived from what was asked for, never restated in config: the flag is
                # how torchmetrics is told to produce a per-class reading, and a run that
                # wants none should not pay for one.
                class_metrics=bool(PER_CLASS_READINGS & set(readings)),
                **kwargs,
            )
        )
        self.readings = tuple(readings)
        self._refuse_unknown_readings()

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        if not isinstance(predictions, Instances) or not isinstance(target, Instances):
            raise TypeError(
                f"'map' compares detected objects, but was given {type(predictions).__name__} "
                f"against {type(target).__name__}. It belongs on a task whose preset is 'detection'."
            )
        # Over the images either side mentions, not each side's own count: a model that
        # found nothing carries no `sample_index` at all, and torchmetrics compares two
        # lists of equal length. Beyond that count both sides are empty and contribute
        # nothing, so this is the comparison rather than a compensation for one.
        images = max(_images_in(predictions), _images_in(target))
        self.inner.update(_per_image(predictions, images), _per_image(target, images))

    def compute(self) -> dict[str, Tensor | PerClass]:
        """The asked-for readings, each in the shape the log grammar can place."""
        found = cast("dict[str, Tensor]", self.inner.compute())
        classes = found[_INDEX]
        published: dict[str, Tensor | PerClass] = {}
        for reading in self.readings:
            value = found[reading]
            if value.ndim == 0 and float(value) == _NOT_APPLICABLE:
                continue
            published[reading] = PerClass(value, classes) if reading in PER_CLASS_READINGS else value
        return published

    def _refuse_unknown_readings(self) -> None:
        """A misspelt reading fails here, not as a key missing from the first epoch's log."""
        unknown = [reading for reading in self.readings if reading not in _COMPUTED_READINGS]
        if unknown:
            raise ValueError(
                f"'map' computes no reading called {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(_COMPUTED_READINGS))}."
            )


_COMPUTED_READINGS: Final = frozenset(
    {
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
        *PER_CLASS_READINGS,
    }
)
"""Every reading ``MeanAveragePrecision.compute`` returns, minus its class index.

Listed rather than read off a computed value, because a misspelt name has to be refused
while the run is being assembled — at which point nothing has been computed yet.
"""


def _images_in(found: Instances) -> int:
    """How many images this side has anything to say about, which may be none at all."""
    return int(found.sample_index.max()) + 1 if len(found.sample_index) else 0


def _per_image(found: Instances, images: int) -> list[dict[str, Tensor]]:
    """The one shape torchmetrics takes: a dict per image, in image order.

    An image that carries nothing still gets its entry — dropped, the images after it
    would shift up and be scored against the wrong annotation.

    Ground truth carries no ``scores``, and none are invented for it: torchmetrics reads
    the key only from the prediction side, so an absent one is the honest answer rather
    than a column of ones that would read as certainty nobody measured.
    """
    entries: list[dict[str, Tensor]] = []
    for index in range(images):
        one = found.of(index)
        entry = {"boxes": one.boxes, "labels": one.labels}
        if one.scores is not None:
            entry["scores"] = one.scores
        entries.append(entry)
    return entries
