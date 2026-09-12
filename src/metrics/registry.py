"""The names a task's ``metrics`` may write."""

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

from src.core import ERROR, OVERLAP, Registry

metric_registry: Registry[Metric] = Registry("metric")
"""What `tasks.<name>.metrics.<label>` writes, under the names a data scientist already uses.

Most entries are torchmetrics classes exactly as they come: a metric that computes a number needs
nothing from us, and wrapping one only to reach it by decorator would add a class that does nothing.
The few readings that mean a *picture* are classes of ours, registered beside their definition —
what a value means is knowledge the library has no place for.

The list is a convenience rather than a gate: anything torchmetrics offers is one ``_target_`` away
without being named here first.
"""

metric_registry.register("accuracy")(Accuracy)
metric_registry.register("f1")(F1Score)
metric_registry.register("precision")(Precision)
metric_registry.register("recall")(Recall)
metric_registry.register(OVERLAP)(JaccardIndex)
metric_registry.register(ERROR)(MeanAbsoluteError)
metric_registry.register("mse")(MeanSquaredError)
