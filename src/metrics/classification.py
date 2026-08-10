"""Classification metrics whose computed value is an artifact rather than a number."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torchmetrics import ROC, ConfusionMatrix, PrecisionRecallCurve

from src.core.reporting import Curve, Matrix
from src.metrics.adapter import WrappedMetric

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torchmetrics import Metric

_BINARY = "binary"
_MULTILABEL = "multilabel"


class ClassificationArtifactMetric(WrappedMetric):
    """A torchmetrics classification metric, sized by the facts its objective offers.

    The three facts are **named** in this signature rather than swept into ``**kwargs``,
    and that is mechanism rather than style: assembly offers derived values to whatever
    *names* them, and ``**kwargs`` names nothing. A wrapper that forwarded blindly would
    be built with no task mode and no class count, then fail inside torchmetrics about an
    argument the config never mentioned. Written once here, so the three below cannot
    drift from one another — and so the one signature that has to be right is one.

    Every objective offers ``task``; only the ones with a class space add a count, and
    which of the two names it carries is the objective's word. All three are handed on,
    because each of the wrapped classes dispatches on ``task`` and ignores the count that
    does not apply to it.

    Subclasses say two things and nothing else: which torchmetrics class does the
    arithmetic (``inner_type``), and what its value means (``compute``).
    """

    inner_type: ClassVar[Callable[..., Metric]]
    higher_is_better = None

    def __init__(
        self,
        task: str,
        num_classes: int | None = None,
        num_labels: int | None = None,
        **kwargs: Any,
    ) -> None:
        build = type(self).inner_type
        super().__init__(build(task=task, num_classes=num_classes, num_labels=num_labels, **kwargs))
        self._task = task


class PrecisionRecallMetric(ClassificationArtifactMetric):
    """The precision-recall curve of a classification task, drawn with recall on x.

    PR and ROC compute the same geometry with the opposite meaning —
    ``(precision, recall, _)`` against ``(fpr, tpr, _)`` — so which tensor is which axis
    is said by the class that knows the metric, never read off the tuple's order. Guessed,
    the two would draw as mirror images of one another, and no chart shows that.
    """

    inner_type = PrecisionRecallCurve

    def compute(self) -> Curve:
        precision, recall, _ = self.inner.compute()
        return Curve(
            x=_lines(recall),
            y=_lines(precision),
            xaxis="Recall",
            yaxis="Precision",
            positive_only=self._task == _BINARY,
        )


class RocMetric(ClassificationArtifactMetric):
    """The receiver operating characteristic of a classification task, with FPR on x.

    The mirror of ``PrecisionRecallMetric``, and the reason both name their axes; see there.
    """

    inner_type = ROC

    def compute(self) -> Curve:
        false_positive, true_positive, _ = self.inner.compute()
        return Curve(
            x=_lines(false_positive),
            y=_lines(true_positive),
            xaxis="FPR",
            yaxis="TPR",
            positive_only=self._task == _BINARY,
        )


class ConfusionMatrixMetric(ClassificationArtifactMetric):
    """The confusion matrix of a classification task, with its axes named.

    A multilabel run computes ``[L, 2, 2]`` — one small matrix per label — which is no
    single plot. Publishing nothing is the honest answer there, and it is settled from the
    declared task rather than from the computed shape: whether this run draws is known
    before the first epoch, and a value that publishes nothing leaves quietly rather than
    warning every epoch about a shape nobody can change.
    """

    inner_type = ConfusionMatrix

    def compute(self) -> Matrix | None:
        if self._task == _MULTILABEL:
            return None
        # Labels left open: the class space belongs to the task, and the router fills it in.
        return Matrix(value=self.inner.compute(), xaxis="Predicted", yaxis="True")


def _lines(values: Any) -> tuple[Tensor, ...]:
    """One tensor per plotted line, whatever shape torchmetrics chose.

    A binary metric computes one tensor and a multiclass one a list per class — or a
    stacked ``[C, points]`` tensor where the thresholds were pinned, since equal-length
    series can share one array.
    """
    if isinstance(values, list):
        return tuple(values)
    if isinstance(values, torch.Tensor) and values.ndim == 2:
        return tuple(values)
    return (values,)
