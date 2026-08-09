"""Registries of the metrics capability."""

from __future__ import annotations

from torchmetrics import (
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    Precision,
    PrecisionRecallCurve,
    Recall,
)

from src.core.registry import Registry
from src.metrics.detection import MeanAveragePrecisionOverInstances

metric_registry: Registry[Metric] = Registry("metric")
"""Config-facing metrics, under the names a data scientist already uses.

Third-party ``torchmetrics`` classes rather than abstractions of our own, so they are
registered explicitly below instead of by decorator. A convenience, not a gate: anything
torchmetrics offers is reachable by ``_target_`` without being registered first.

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
metric_registry.register("confusion_matrix")(ConfusionMatrix)
metric_registry.register("precision_recall_curve")(PrecisionRecallCurve)
metric_registry.register("roc")(ROC)
metric_registry.register("map")(MeanAveragePrecisionOverInstances)
