"""``WrappedMetricSet``: named torchmetrics behind the ``MetricSet`` port."""

from __future__ import annotations

import pytest
import torch
from torchmetrics import Accuracy, Metric

from src.metrics import WrappedMetricSet


def make_metrics() -> WrappedMetricSet:
    return WrappedMetricSet({"accuracy": Accuracy(task="multiclass", num_classes=3)})


def test_update_accumulates_and_compute_reports_by_name() -> None:
    metrics = make_metrics()
    metrics.update(torch.tensor([0, 1]), torch.tensor([0, 2]))

    computed = metrics.compute()

    assert computed["accuracy"].item() == pytest.approx(0.5)


def test_reset_clears_the_accumulated_state() -> None:
    metrics = make_metrics()
    metrics.update(torch.tensor([0]), torch.tensor([1]))
    metrics.reset()
    metrics.update(torch.tensor([0]), torch.tensor([0]))

    assert metrics.compute()["accuracy"].item() == pytest.approx(1.0)


def test_directions_come_from_torchmetrics() -> None:
    assert make_metrics().directions() == {"accuracy": True}


def test_wrapped_metrics_are_registered_for_device_movement() -> None:
    metrics = make_metrics()

    assert any(isinstance(module, Metric) for module in metrics.modules())
