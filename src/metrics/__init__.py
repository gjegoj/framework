"""What a run is judged by: torchmetrics under the names a declaration writes, plus the readings that draw.

A metric set is a ``torchmetrics.MetricCollection`` and nothing else — it already accumulates over
batches, groups the metrics that share state, moves with the model and resets. A second collection API
of ours would only forward to it. What this package adds is the naming: which names a config may write,
what sizes them, and what a value means when it is not a number.

Importing this package is what makes those names resolvable: `metrics: {f1: {name: f1}}` finds one here.

``MetricCollection`` is published here as well, unchanged: it is what a build hands back and what a
training loop keeps per stage, so the one door to torchmetrics is this package rather than an import
of the library in every consumer.
"""

from __future__ import annotations

from torchmetrics import MetricCollection

from src.metrics.classification import ConfusionMatrix
from src.metrics.metric_learning import RecallAtK

__all__ = ["ConfusionMatrix", "MetricCollection", "RecallAtK"]
