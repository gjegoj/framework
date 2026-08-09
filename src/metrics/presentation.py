"""What a computed value *means*, keyed by metric class — an open table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
    MultilabelConfusionMatrix,
    MultilabelPrecisionRecallCurve,
    MultilabelROC,
)

from src.core.entities import Curve, Matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torchmetrics import Metric

type _Translate = Callable[[Any], Any]

_PRESENTATIONS: dict[type, _Translate] = {}


def presentation_of(*metric_types: type) -> Callable[[_Translate], _Translate]:
    """Register how a metric class's computed value is drawn.

    The translator receives the raw computed value and returns a drawable
    artifact (``Curve``, ``Matrix``); ``None`` for "identified, not drawable"
    — dropped quietly instead of warned about every epoch; or
    ``NotImplemented`` for "not my geometry" — the MRO walk continues.

    The last case is what makes inheritance safe: torchmetrics subclasses for
    *state reuse* (``JaccardIndex`` extends ``ConfusionMatrix``, ``AUROC``
    extends the PR curve) while changing what ``compute`` returns, so a
    translator must vouch for the value's geometry before claiming it.
    """

    def register(translate: _Translate) -> _Translate:
        for metric_type in metric_types:
            _PRESENTATIONS[metric_type] = translate
        return translate

    return register


def present(metric: Metric, value: Any) -> Any:
    """The metric's value as its family draws it; untouched when no family claims it."""
    for cls in type(metric).__mro__:
        translate = _PRESENTATIONS.get(cls)
        if translate is not None:
            presented = translate(value)
            if presented is not NotImplemented:
                return presented
    return value


def _curve_tuple(value: Any) -> bool:
    """The PR/ROC compute shape — a scalar descendant (AUROC) computes something else."""
    return isinstance(value, tuple) and len(value) == 3


@presentation_of(BinaryPrecisionRecallCurve)
def _binary_pr(value: Any) -> Any:
    if not _curve_tuple(value):
        return NotImplemented
    precision, recall, _ = value
    return Curve(x=(recall,), y=(precision,), xaxis="Recall", yaxis="Precision", positive_only=True)


@presentation_of(MulticlassPrecisionRecallCurve, MultilabelPrecisionRecallCurve)
def _pr(value: Any) -> Any:
    if not _curve_tuple(value):
        return NotImplemented
    precision, recall, _ = value
    return Curve(x=_lines(recall), y=_lines(precision), xaxis="Recall", yaxis="Precision")


@presentation_of(BinaryROC)
def _binary_roc(value: Any) -> Any:
    if not _curve_tuple(value):
        return NotImplemented
    fpr, tpr, _ = value
    return Curve(x=(fpr,), y=(tpr,), xaxis="FPR", yaxis="TPR", positive_only=True)


@presentation_of(MulticlassROC, MultilabelROC)
def _roc(value: Any) -> Any:
    if not _curve_tuple(value):
        return NotImplemented
    fpr, tpr, _ = value
    return Curve(x=_lines(fpr), y=_lines(tpr), xaxis="FPR", yaxis="TPR")


@presentation_of(BinaryConfusionMatrix, MulticlassConfusionMatrix)
def _confusion(value: Any) -> Any:
    if not (isinstance(value, torch.Tensor) and value.ndim == 2):
        return NotImplemented  # A reduced descendant (IoU) no longer carries the matrix.
    return Matrix(value=value, xaxis="Predicted", yaxis="True")


@presentation_of(MultilabelConfusionMatrix)
def _multilabel_confusion(value: Any) -> Any:
    if not (isinstance(value, torch.Tensor) and value.ndim == 3):
        return NotImplemented  # A reduced descendant (multilabel IoU) computes a vector.
    return None  # [L, 2, 2]: no single plot; identified, so no warning either.


def _lines(values: Any) -> tuple[Tensor, ...]:
    """One tensor per plotted line, whatever shape torchmetrics chose.

    Multiclass emits a list per class, or a stacked ``[C, points]`` tensor
    when thresholds were pinned.
    """
    if isinstance(values, list):
        return tuple(values)
    if isinstance(values, torch.Tensor) and values.ndim == 2:
        return tuple(values)
    return (values,)


__all__ = ["present", "presentation_of"]
