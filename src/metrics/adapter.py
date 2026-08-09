"""torchmetrics behind the ``MetricSet`` port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torchmetrics import MetricCollection

from src.core.log_keys import join
from src.core.ports import MetricFamily, MetricSet
from src.metrics.presentation import present

if TYPE_CHECKING:
    from torchmetrics import Metric

    from src.core.entities import TaskOutput


class WrappedMetricSet(MetricSet):
    """A named collection of torchmetrics — the same convention as ``WrappedCriterion``.

    Backed by a ``MetricCollection``: metrics sharing internal state
    (precision/recall/f1 over one confusion matrix) are updated once per
    group, and the collection registers as a submodule, so state moves
    across devices with the model. ``directions`` reads each metric's own
    ``higher_is_better`` flag.
    """

    def __init__(self, metrics: Mapping[str, Metric]) -> None:
        super().__init__()
        self.collection = MetricCollection(dict(metrics))

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        self.collection.update(predictions, target)

    def compute(self) -> dict[str, Any]:
        # Per member rather than `self.collection.compute()`, which flattens a metric
        # whose value is a *family* of readings up to the top level and drops the label
        # it was registered under — after which the lookup below misses, and two entries
        # of one metric collide on identical keys. Measured, the values are identical:
        # compute groups do their saving during `update`.
        #
        # This is also the last place that knows *which* metric produced a value, so
        # presentation happens here, and "identified but not drawable" leaves the dict.
        presented = {label: present(metric, metric.compute()) for label, metric in self.collection.items()}
        return {label: value for label, value in presented.items() if value is not None}

    def reset(self) -> None:
        self.collection.reset()

    def directions(self) -> dict[str, bool | None]:
        """Which way is better, keyed exactly as each value is logged.

        A metric publishing a family of readings answers for each of them: they are
        separate keys in the log, so a checkpoint watching ``map_50`` needs its own
        entry — naming only the metric would leave every reading but one unwatchable.
        """
        return {
            key: metric.higher_is_better
            for label, metric in self.collection.items()
            for key in _published_by(label, metric)
        }


def _published_by(label: str, metric: Metric) -> list[str]:
    """The keys one metric writes under: its label, or a leaf per reading of a family.

    Asked of the metric rather than of a computed value, because directions are read
    before a run starts — a checkpoint monitor is configured at assembly, not at the
    first epoch.
    """
    if not isinstance(metric, MetricFamily):
        return [label]
    return [join(label, str(reading)) for reading in metric.readings]
