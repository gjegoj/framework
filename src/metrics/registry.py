"""Registries of the metrics capability."""

from __future__ import annotations

from torchmetrics import (
    Accuracy,
    F1Score,
    JaccardIndex,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    Precision,
    Recall,
)

from src.core.registry import Registry
from src.metrics.classification import ConfusionMatrixMetric, PrecisionRecallMetric, RocMetric
from src.metrics.detection import MeanAveragePrecisionOverInstances

metric_registry: Registry[Metric] = Registry("metric")
"""Config-facing metrics, under the names a data scientist already uses.

Registered here rather than by decorator, ours included, so one file lists every name a
config may write. A convenience, not a gate: anything torchmetrics offers is reachable by
``_target_`` without being registered first.

Most entries are torchmetrics classes as they come, because a metric that computes a
number needs nothing from us. The few that compute an **artifact** — a curve, a matrix —
are classes of ours instead, since what a value means is knowledge torchmetrics has no
place for. That is the whole of the distinction; see ``WrappedMetric``.

An objective's own ``metric_kwargs`` (``task``, ``num_classes`` / ``num_labels``) merge
with each entry's declared params when the sets are built.
"""

metric_registry.register("accuracy")(Accuracy)
metric_registry.register("f1")(F1Score)
metric_registry.register("precision")(Precision)
metric_registry.register("recall")(Recall)
metric_registry.register("iou")(JaccardIndex)
metric_registry.register("mae")(MeanAbsoluteError)
metric_registry.register("mse")(MeanSquaredError)
metric_registry.register("confusion_matrix")(ConfusionMatrixMetric)
metric_registry.register("precision_recall_curve")(PrecisionRecallMetric)
metric_registry.register("roc")(RocMetric)
metric_registry.register("map")(MeanAveragePrecisionOverInstances)
