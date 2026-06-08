"""Registry of torchmetrics classes selectable from YAML by name.

Maps short config keys (``accuracy``, ``f1``, ...) to torchmetrics classes so a
``metrics:`` block can pick and parametrise them (``top_k``, ``average``, ...)
without code. Users register their own metric the same way, or bypass the
registry entirely with a ``_target_`` spec.
"""

from __future__ import annotations

from dataclasses import dataclass

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
metric_factories.register("jaccard")(tm.JaccardIndex)
metric_factories.register("iou")(tm.JaccardIndex)
metric_factories.register("mse")(tm.MeanSquaredError)
metric_factories.register("mae")(tm.MeanAbsoluteError)
metric_factories.register("precision_recall_curve")(tm.PrecisionRecallCurve)
metric_factories.register("roc")(tm.ROC)


@dataclass(frozen=True)
class CurveSpec:
    """Axis labels and output-index mapping for a torchmetrics curve metric.

    torchmetrics curve metrics return ``(first, second, thresholds)``, but the
    semantic order differs across families:
      - PrecisionRecallCurve: ``(precision, recall, ...)`` — recall is index 1
      - ROC / DET:            ``(fpr, tpr/fnr, ...)``     — fpr is index 0

    ``x_idx`` and ``y_idx`` select which element (0 or 1) maps to each axis so
    the handler can extract the correct tensors regardless of metric family.

    Parameters:
        xaxis (str): X-axis label shown in the plot backend.
        yaxis (str): Y-axis label shown in the plot backend.
        x_idx (int): Index into the tuple that provides X values.
        y_idx (int): Index into the tuple that provides Y values.
    """

    xaxis: str
    yaxis: str
    x_idx: int = 1
    y_idx: int = 0


# Default axis specs consumed by MatrixMetricHandler and CurveMetricHandler.
# Key matches the metric_factories registration name above.

matrix_axes: dict[str, tuple[str, str]] = {
    "confusion_matrix": ("Predicted", "True"),
}

curve_specs: dict[str, CurveSpec] = {
    "precision_recall_curve": CurveSpec(xaxis="Recall", yaxis="Precision", x_idx=1, y_idx=0),
    "roc": CurveSpec(xaxis="FPR", yaxis="TPR", x_idx=0, y_idx=1),
    "det_curve": CurveSpec(xaxis="FPR", yaxis="FNR", x_idx=0, y_idx=1),
}
