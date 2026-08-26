"""torchmetrics behind the framework's ports — one metric, and a named set of them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, override

from torchmetrics import Metric, MetricCollection

from src.core.log_keys import join
from src.core.ports import MetricSet, MultiReadingMetric

if TYPE_CHECKING:
    from src.core.entities import TaskOutput


class WrappedMetric(Metric):
    """A torchmetrics metric behind a class of ours — the "wrap a module" rule the criteria follow.

    A subclass says what its computed value *means* by returning an artifact from
    ``compute`` — a ``Curve``, a ``Matrix``, a ``PerClass`` — or ``None`` to publish
    nothing. Meaning lives on a class of ours because torchmetrics subclasses for state
    reuse while changing what ``compute`` returns (``JaccardIndex`` extends
    ``ConfusionMatrix``), so nothing keyed on its hierarchy is safe.

    Parameters:
        inner (Metric): The torchmetrics metric doing the arithmetic; a child module, so
            its state moves across devices with the model.
    """

    full_state_update = False

    def __init__(self, inner: Metric) -> None:
        super().__init__()
        self.inner = inner

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        self.inner.update(predictions, target)

    def reset(self) -> None:
        self.inner.reset()
        super().reset()


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

    @override
    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        self.collection.update(predictions, target)

    @override
    def compute(self) -> dict[str, Any]:
        # Per member rather than `self.collection.compute()`, which flattens a metric
        # whose value is a *family* of readings up to the top level and drops the label
        # it was registered under, after which two entries of one metric collide on
        # identical keys. Measured, the values are identical: compute groups do their
        # saving during `update`.
        computed = {label: metric.compute() for label, metric in self.collection.items()}
        # `None` is a metric saying its value is identified and draws as nothing — the
        # multilabel confusion matrix. It leaves here rather than reaching the router,
        # which would have no way to tell it from an artifact nobody recognised.
        return {label: value for label, value in computed.items() if value is not None}

    @override
    def reset(self) -> None:
        self.collection.reset()

    @override
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
    if not isinstance(metric, MultiReadingMetric):
        return [label]
    return [join(label, str(reading)) for reading in metric.readings]
