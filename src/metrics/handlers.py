"""Metric handler chain — Strategy / CoR over ``MetricSet.compute()`` outputs.

Shape → handler mapping:
  0-D scalar  → ScalarMetricHandler
  1-D vector  → VectorMetricHandler  (per-class, average=none)
  2-D matrix  → MatrixMetricHandler  (confusion matrix)
  3-tuple     → CurveMetricHandler   (PR curve, ROC, ...)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from src.core.keys import MEAN
from src.core.ports import CurveLogger, MatrixLogger


@dataclass(frozen=True)
class MetricLogContext:
    """Carries everything a handler needs for one metric value.

    Parameters:
        log_scalar (Callable): Bound ``self.log`` from ``LitModule``.
        logger (object): Active logger; plot handlers narrow it to the artifact port they need at use.
        step (int): Current epoch, used as iteration counter for plot backends.
        class_names (list[str] | None): Maps class index → display name.
            ``None`` falls back to ``class{i}`` keys.
        metric_name (str | None): Registry key of the current metric, used to
            look up axis labels in ``MatrixMetricHandler`` and ``CurveMetricHandler``.
    """

    log_scalar: Callable[[str, Any], None]
    logger: object
    step: int
    class_names: list[str] | None = None
    metric_name: str | None = None


@dataclass(frozen=True)
class CurveSpec:
    """Axis labels and output-index mapping for a torchmetrics curve metric.

    torchmetrics curve metrics return ``(first, second, thresholds)``, but the
    semantic order differs across families:
      - PrecisionRecallCurve: ``(precision, recall, ...)`` — recall is index 1
      - ROC / DET:            ``(fpr, tpr/fnr, ...)``     — fpr is index 0

    ``x_index`` and ``y_index`` select which element (0 or 1) maps to each axis so the
    handler can extract the correct tensors regardless of metric family.

    Parameters:
        xaxis (str): X-axis label shown in the plot backend.
        yaxis (str): Y-axis label shown in the plot backend.
        x_index (int): Index into the tuple that provides X values.
        y_index (int): Index into the tuple that provides Y values.
    """

    xaxis: str
    yaxis: str
    x_index: int = 1
    y_index: int = 0


class MetricHandler(ABC):
    """One link in the metric-logging chain."""

    @abstractmethod
    def can_handle(self, value: Any) -> bool: ...

    @abstractmethod
    def handle(self, key: str, value: Any, context: MetricLogContext) -> None: ...


class ScalarMetricHandler(MetricHandler):
    def can_handle(self, value: Any) -> bool:
        if isinstance(value, tuple):
            return False
        return not isinstance(value, torch.Tensor) or value.ndim == 0

    def handle(self, key: str, value: Any, context: MetricLogContext) -> None:
        context.log_scalar(key, value)


class VectorMetricHandler(MetricHandler):
    """Logs mean at ``key/mean`` and each class at ``key/<class_name>``.

    Grouping mean and per-class values under the same key prefix places them on
    the same plot in backends like ClearML.
    """

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim == 1

    def handle(self, key: str, value: Any, context: MetricLogContext) -> None:
        context.log_scalar(f"{key}/{MEAN}", value.float().mean())
        for i, class_value in enumerate(value):
            context.log_scalar(f"{key}/{_class_label(i, context.class_names)}", class_value.float())


class MatrixMetricHandler(MetricHandler):
    """Logs 2-D tensors via ``MatrixLogger.log_matrix``.

    Silently skips when the active logger does not implement ``MatrixLogger``.

    Parameters:
        axes (dict | None): ``{metric_name: (xaxis_label, yaxis_label)}``.
    """

    def __init__(self, axes: dict[str, tuple[str, str]] | None = None) -> None:
        self._axes = axes or {}

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim == 2

    def handle(self, key: str, value: Any, context: MetricLogContext) -> None:
        if not isinstance(context.logger, MatrixLogger):
            return
        xaxis, yaxis = self._axes.get(context.metric_name or "", (None, None))
        context.logger.log_matrix(
            title=key,
            matrix=value,
            iteration=context.step,
            labels=context.class_names,
            xaxis=xaxis,
            yaxis=yaxis,
        )


class CurveMetricHandler(MetricHandler):
    """Logs 3-tuple curve metrics via ``PlotLogger.log_curve``.

    torchmetrics curve metrics return ``(first, second, thresholds)`` where
    ``first`` and ``second`` are lists of tensors (multiclass) or single tensors
    (binary). ``CurveSpec.x_index`` / ``y_index`` select which tuple element is X vs Y
    — necessary because metric families disagree on ordering (PR: precision first;
    ROC: fpr first).

    Silently skips when the active logger does not implement ``CurveLogger``.

    Parameters:
        specs (dict | None): ``{metric_name: CurveSpec}``.
    """

    def __init__(self, specs: dict[str, CurveSpec] | None = None) -> None:
        self._specs = specs or {}

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, tuple) and len(value) == 3

    def handle(self, key: str, value: Any, context: MetricLogContext) -> None:
        if not isinstance(context.logger, CurveLogger):
            return
        spec = self._specs.get(context.metric_name or "", CurveSpec(xaxis="x", yaxis="y"))
        first_output, second_output, _ = value
        is_binary = not isinstance(first_output, list)  # binary metrics emit single tensors
        x_per_class = _to_per_class_list(first_output if spec.x_index == 0 else second_output)
        y_per_class = _to_per_class_list(first_output if spec.y_index == 0 else second_output)
        for i, (x_values, y_values) in enumerate(zip(x_per_class, y_per_class)):
            context.logger.log_curve(
                title=key,
                x=x_values,
                y=y_values,
                iteration=context.step,
                series=_curve_series(i, context.class_names, is_binary),
                xaxis=spec.xaxis,
                yaxis=spec.yaxis,
            )


def _to_per_class_list(output: Any) -> list[Any]:
    return output if isinstance(output, list) else [output]


def _class_label(index: int, class_names: list[str] | None) -> str:
    if class_names and index < len(class_names):
        return class_names[index]
    return f"class{index}"


def _curve_series(index: int, class_names: list[str] | None, is_binary: bool) -> str:
    """Series label for one curve. A binary metric has one curve for the *positive*
    class (index 1), so labelling it with index 0 would mislabel it as the negative class."""
    if is_binary:
        return class_names[1] if class_names and len(class_names) >= 2 else "positive"
    return _class_label(index, class_names)


# Default axis specs for the matrix / curve handlers, keyed by the metric_factories
# registration name (``confusion_matrix``, ``precision_recall_curve``, ...).
MATRIX_AXES: dict[str, tuple[str, str]] = {
    "confusion_matrix": ("Predicted", "True"),
}

CURVE_SPECS: dict[str, CurveSpec] = {
    "precision_recall_curve": CurveSpec(xaxis="Recall", yaxis="Precision", x_index=1, y_index=0),
    "roc": CurveSpec(xaxis="FPR", yaxis="TPR", x_index=0, y_index=1),
}


DEFAULT_METRIC_HANDLERS: tuple[MetricHandler, ...] = (
    ScalarMetricHandler(),
    VectorMetricHandler(),
    MatrixMetricHandler(MATRIX_AXES),
    CurveMetricHandler(CURVE_SPECS),
)


def dispatch(
    key: str,
    value: Any,
    context: MetricLogContext,
    handlers: tuple[MetricHandler, ...] = DEFAULT_METRIC_HANDLERS,
) -> None:
    """Route ``value`` through the handler chain; warn and skip if nothing matches."""
    for handler in handlers:
        if handler.can_handle(value):
            handler.handle(key, value, context)
            return
    if isinstance(value, torch.Tensor):
        warnings.warn(f"No handler for metric '{key}' with ndim={value.ndim}; skipping.", stacklevel=2)
