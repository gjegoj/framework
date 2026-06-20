"""Selection of a stage's headline metrics for an end-of-run summary.

Kept as a free function (testable without Lightning), mirroring ``directions.py``.
``MetricSummaryCallback`` feeds the result to ``PlotLogger.log_single_value`` so a
backend like ClearML renders a compact "Single Values" table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.core.keys import LOSS, MEAN, TOTAL


def summary_metrics(metrics: Mapping[str, Any], stage: str) -> dict[str, float]:
    """Pick the headline scalar metrics for ``stage``, keyed as the live table names them.

    Keeps, for the given stage: scalar task metrics ``task/metric/{stage}``, each vector
    metric's averaged value ``task/metric/{stage}/mean``, and the aggregate loss
    ``loss/{stage}/total``. Drops per-class vector leaves and per-component losses. The
    stage and the ``mean`` leaf are stripped from the key, so names match the training
    table's rows (``species/f1``, ``breed/recall``, ``mask/iou``, ``loss/total``).

    Parameters:
        metrics (Mapping[str, Any]): Logged values keyed ``task/metric/stage`` etc.
            (e.g. ``trainer.callback_metrics``); values convertible to ``float``.
        stage (str): Lifecycle stage to summarize (e.g. ``"test"``).

    Returns:
        dict[str, float]: ``{display_name: value}`` headline metrics, in input order.
    """
    selected: dict[str, float] = {}
    for key, value in metrics.items():
        parts = key.split("/")
        if parts[0] == LOSS:
            if len(parts) == 3 and parts[1] == stage and parts[2] == TOTAL:
                selected[f"{parts[0]}/{parts[2]}"] = float(value)  # loss/<stage>/total -> loss/total
        elif len(parts) == 3 and parts[2] == stage:
            selected[f"{parts[0]}/{parts[1]}"] = float(value)  # task/metric/<stage> -> task/metric
        elif len(parts) == 4 and parts[2] == stage and parts[3] == MEAN:
            selected[f"{parts[0]}/{parts[1]}"] = float(value)  # task/metric/<stage>/mean -> task/metric
    return selected
