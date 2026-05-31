"""Registry of torchmetrics classes selectable from YAML by name.

Maps short config keys (``accuracy``, ``f1``, ...) to torchmetrics classes so a
``metrics:`` block can pick and parametrise them (``top_k``, ``average``, ...)
without code. Users register their own metric the same way, or bypass the
registry entirely with a ``_target_`` spec.
"""

from __future__ import annotations

import torchmetrics as tm

from src.core.registry import Registry

metric_factories: Registry[tm.Metric] = Registry("metric")

metric_factories.register("accuracy")(tm.Accuracy)
metric_factories.register("precision")(tm.Precision)
metric_factories.register("recall")(tm.Recall)
metric_factories.register("f1")(tm.F1Score)
metric_factories.register("auroc")(tm.AUROC)
metric_factories.register("average_precision")(tm.AveragePrecision)
metric_factories.register("confusion_matrix")(tm.ConfusionMatrix)
metric_factories.register("mse")(tm.MeanSquaredError)
metric_factories.register("mae")(tm.MeanAbsoluteError)
