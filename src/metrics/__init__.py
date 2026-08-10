"""The metrics capability: torchmetrics behind the core ``MetricSet`` port."""

from __future__ import annotations

from src.metrics.adapter import WrappedMetric, WrappedMetricSet

__all__ = ["WrappedMetric", "WrappedMetricSet"]
