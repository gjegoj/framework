"""Projection of a task collection into per-metric optimization directions.

Kept as a free function (not a ``LitModule`` method body) so the logic is unit
-testable without Lightning. ``LitModule`` exposes it via a one-line accessor,
keeping the module thin while owning its own metric-key convention.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.core.entities import Task


def task_metric_directions(tasks: Iterable[Task]) -> dict[str, bool | None]:
    """Map each logged metric key to its metric's ``higher_is_better`` flag.

    Keys are built with the same ``task/metric/stage`` convention the training
    module logs under, so a consumer can bind direction to a metric without
    re-deriving it from the metric's name.

    Parameters:
        tasks (Iterable[Task]): The tasks whose metric sets to project.

    Returns:
        dict[str, bool | None]: ``{f"{task}/{metric}/{stage}": higher_is_better}``.
            ``True`` = larger is better, ``False`` = smaller is better,
            ``None`` = no direction (confusion matrix, curves).
    """
    directions: dict[str, bool | None] = {}
    for task in tasks:
        for stage, metric_set in task.metrics.items():
            for name, higher_is_better in metric_set.directions().items():
                directions[f"{task.name}/{name}/{stage}"] = higher_is_better
    return directions
