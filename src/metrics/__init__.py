"""What a run is judged by: torchmetrics under the names a declaration writes, plus the readings that draw.

A metric set is a ``torchmetrics.MetricCollection`` and nothing else — it already accumulates over
batches, groups the metrics that share state, moves with the model and resets. A second collection API
of ours would only forward to it. What this package adds is the naming: which names a config may write,
what sizes them, and what a value means when it is not a number.

Importing this package is what makes those names resolvable: `metrics: {f1: {name: f1}}` finds one here.
"""

from __future__ import annotations

from src.metrics.classification import ConfusionMatrix

__all__ = ["ConfusionMatrix"]
