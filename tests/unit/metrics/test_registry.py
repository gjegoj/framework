"""The metric factories registry: torchmetrics under config-friendly names."""

from __future__ import annotations

import torch
from torchmetrics import Metric

from src.metrics.registry import metric_registry


def test_built_in_factories_are_registered() -> None:
    assert set(metric_registry) == {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "iou",
        "mae",
        "mse",
        "confusion_matrix",
        "precision_recall_curve",
        "roc",
        "map",
    }


def test_create_builds_a_configured_metric() -> None:
    metric = metric_registry.create("accuracy", task="multiclass", num_classes=3)

    metric.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    value = metric.compute()

    assert isinstance(metric, Metric)
    assert value.item() == 1.0
