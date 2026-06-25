"""Selection of a stage's headline metrics for an end-of-run summary.

Kept as a free function (testable without Lightning), mirroring ``directions.py``.
``MetricSummaryCallback`` feeds the result to ``PlotLogger.log_single_value`` so a
backend like ClearML renders a compact "Single Values" table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.core.keys import MEAN, TOTAL
from src.core.metric_key import MetricKey


def summary_metrics(metrics: Mapping[str, Any], stage: str) -> dict[str, float]:
    """Pick the headline scalar metrics for ``stage``, keyed as the live table names them.

    Keeps, for the given stage: scalar task metrics ``task/metric/{stage}``, each vector
    metric's averaged value ``task/metric/{stage}/mean``, and the aggregate loss
    ``loss/{stage}/total``. Drops per-class vector leaves and per-component losses. The
    key grammar is parsed once by :class:`MetricKey`; ``display_name`` strips the stage
    and the ``mean`` leaf, so names match the training table's rows (``species/f1``,
    ``breed/recall``, ``mask/iou``, ``loss/total``).

    Parameters:
        metrics (Mapping[str, Any]): Logged values keyed ``task/metric/stage`` etc.
            (e.g. ``trainer.callback_metrics``); values convertible to ``float``.
        stage (str): Lifecycle stage to summarize (e.g. ``"test"``).

    Returns:
        dict[str, float]: ``{display_name: value}`` headline metrics, in input order.
    """
    selected: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = MetricKey.parse(key)
        if metric_key.stage != stage:
            continue
        if metric_key.is_loss:
            if metric_key.name == TOTAL:  # aggregate loss only; drop per-component terms
                selected[metric_key.display_name] = float(value)
        elif metric_key.leaf is None or metric_key.leaf == MEAN:  # scalar metric or vector mean
            selected[metric_key.display_name] = float(value)
    return selected
