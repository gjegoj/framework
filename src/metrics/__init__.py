"""Use torchmetrics.Metric and MetricCollection directly, once per task/stage/split.

Task prepares metric arguments; Lightning owns update, compute and reset.
Tracking formats scalar and artifact results. There is no second MetricCollection API.
"""

from __future__ import annotations
