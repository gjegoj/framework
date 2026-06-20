"""MetricReporter: log a task's computed metrics, routing each value by its shape.

Owns the epoch-end lifecycle — compute the metric set, route every value through the
handler chain (Chain of Responsibility), then reset — and holds the handler chain
itself. The training module delegates here so it stays a thin Lightning humble object:
it provides only what it owns (the bound ``log`` sink, the active logger, the epoch),
not the metric-logging mechanics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from src.core.entities import Task
from src.core.enums import Stage
from src.metrics.handlers import DEFAULT_METRIC_HANDLERS, MetricHandler, MetricLogContext, dispatch


class MetricReporter:
    """Computes, logs (by metric shape), and resets task metric sets at epoch end.

    Parameters:
        handlers (Sequence[MetricHandler]): Ordered handler chain; the first whose
            ``can_handle`` accepts a value handles it. Defaults to the
            scalar/vector/matrix/curve chain.
    """

    def __init__(self, handlers: Sequence[MetricHandler] = DEFAULT_METRIC_HANDLERS) -> None:
        self._handlers = tuple(handlers)

    def report(
        self,
        task: Task,
        stage: Stage,
        log_scalar: Callable[[str, Any], None],
        logger: object,
        step: int,
    ) -> None:
        """Compute ``task``'s ``stage`` metrics, log each under ``task/metric/stage``, then reset.

        The constant log context (sink/logger/step/class-names) is built once; only
        ``metric_name`` varies per metric.

        Parameters:
            task (Task): Task whose metric set to report.
            stage (Stage): Lifecycle stage selecting the metric set.
            log_scalar (Callable[[str, Any], None]): Sink for scalar values (bound ``self.log``).
            logger (object): Active logger; plot handlers use it only if it is a ``PlotLogger``.
            step (int): Iteration counter for plot backends (the current epoch).
        """
        metric_set = task.metrics[stage]
        context = MetricLogContext(log_scalar=log_scalar, logger=logger, step=step, class_names=task.class_names)
        for name, value in metric_set.compute().items():
            dispatch(f"{task.name}/{name}/{stage}", value, replace(context, metric_name=name), self._handlers)
        metric_set.reset()
