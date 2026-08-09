"""The metrics capability: torchmetrics behind the core ``MetricSet`` port."""

from __future__ import annotations

from src.metrics.adapter import WrappedMetricSet

__all__ = ["WrappedMetricSet"]
