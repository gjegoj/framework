"""Task-layer codecs: light shape/type adaptation of targets for loss/metrics.

Heavy decoding (label -> index, mask I/O) happens in the data layer; here we
only adjust the already-tensor target to the shape each objective's criterion
and metrics expect.
"""

from __future__ import annotations

from torch import Tensor

from src.core.entities import TargetView
from src.core.ports import TaskCodec


class MulticlassTaskCodec(TaskCodec):
    """Class-index targets for cross-entropy and accuracy: ``[B]`` long tensor.

    Also accepts *soft* targets ``[B, C]`` float (produced by MixUp/CutMix): the
    loss keeps the soft distribution (``nn.CrossEntropyLoss`` consumes it
    natively) while the metric target collapses to the dominant class via argmax.
    """

    def adapt(self, target: Tensor) -> TargetView:
        if target.is_floating_point() and target.ndim == 2:
            return TargetView(loss=target, metric=target.argmax(dim=-1))
        if target.ndim == 2 and target.size(-1) == 1:
            target = target.squeeze(-1)
        target = target.long()
        return TargetView(loss=target, metric=target)


class BinaryTaskCodec(TaskCodec):
    """Binary target: ``[B, 1]`` float for BCE loss; ``[B, 1]`` long for metrics.

    BCEWithLogitsLoss needs float targets matching the logit shape ``[B, 1]``.
    torchmetrics binary metrics require preds and targets to have the same
    shape — since logits/predictions are ``[B, 1]``, metric target stays
    ``[B, 1]`` as well (torchmetrics treats ``[N, 1]`` the same as ``[N]``
    for binary tasks as long as both shapes match).
    """

    def adapt(self, target: Tensor) -> TargetView:
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        loss_target = target.float()
        metric_target = target.long()
        return TargetView(loss=loss_target, metric=metric_target)


class MultilabelTaskCodec(TaskCodec):
    """Multilabel target: ``[B, C]`` float for BCE; ``[B, C]`` long for metrics."""

    def adapt(self, target: Tensor) -> TargetView:
        loss_target = target.float()
        metric_target = target.long()
        return TargetView(loss=loss_target, metric=metric_target)


class ContinuousTaskCodec(TaskCodec):
    """Continuous (regression) target: ``[B, 1]`` float for both loss and metrics."""

    def adapt(self, target: Tensor) -> TargetView:
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        target = target.float()
        return TargetView(loss=target, metric=target)
