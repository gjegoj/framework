"""Builders that assemble torchmetrics collections into ``MetricSet`` instances.

Objective strategies call these to produce per-stage metrics sized from the
runtime class count.
"""

from __future__ import annotations

from typing import Literal

from torchmetrics import Accuracy, MetricCollection

from src.core.ports import MetricSet
from src.metrics.adapter import TorchMetricsAdapter

ClassificationTask = Literal["binary", "multiclass", "multilabel"]


def build_classification_metrics(num_classes: int, task: ClassificationTask) -> MetricSet:
    """Build a classification metric set (accuracy) for the given torchmetrics task.

    Parameters:
        num_classes (int): Number of classes (or labels for multilabel).
        task (str): torchmetrics task — ``"binary"``/``"multiclass"``/``"multilabel"``.

    Returns:
        MetricSet: A fresh metric set (new state) for one stage.
    """
    collection = MetricCollection(
        {"accuracy": Accuracy(task=task, num_classes=num_classes)},
    )
    return TorchMetricsAdapter(collection)
