"""Task-layer target adapters: light shape/type adaptation of targets for loss/metrics.

Heavy encoding (label -> index, mask I/O) happens in the data layer (``data/encoders.py``);
here we only adjust the already-tensor target to the shape each objective's criterion and
metrics expect.
"""

from __future__ import annotations

from torch import Tensor

from src.core.entities import TargetView
from src.core.ports import TargetAdapter


class MulticlassTargetAdapter(TargetAdapter):
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


class BinaryTargetAdapter(TargetAdapter):
    """Binary target: ``[B, 1]`` float for BCE loss; ``[B, 1]`` long for metrics.

    BCEWithLogitsLoss needs float targets matching the single-logit shape ``[B, 1]``.
    torchmetrics binary metrics require preds and targets to have the same shape — since
    logits/predictions are ``[B, 1]``, the metric target stays ``[B, 1]`` (torchmetrics
    treats ``[N, 1]`` the same as ``[N]`` for binary as long as both shapes match).

    Also accepts *soft* targets ``[B, 2]`` float (one-hot then mixed by MixUp/CutMix): the
    loss keeps the soft probability of the positive class (index 1) as ``[B, 1]``, while the
    metric target collapses to the dominant class via argmax.
    """

    def adapt(self, target: Tensor) -> TargetView:
        if target.is_floating_point() and target.ndim == 2 and target.size(-1) == 2:
            positive = target[:, 1:2]  # soft P(positive) for the single-logit head, [B, 1]
            return TargetView(loss=positive, metric=target.argmax(dim=-1, keepdim=True))
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        return TargetView(loss=target.float(), metric=target.long())


class MultilabelTargetAdapter(TargetAdapter):
    """Multilabel target: ``[B, C]`` float for BCE; ``[B, C]`` long for metrics."""

    def adapt(self, target: Tensor) -> TargetView:
        loss_target = target.float()
        metric_target = target.long()
        return TargetView(loss=loss_target, metric=metric_target)


class ContinuousTargetAdapter(TargetAdapter):
    """Continuous (regression) target: ``[B, 1]`` float for both loss and metrics."""

    def adapt(self, target: Tensor) -> TargetView:
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        target = target.float()
        return TargetView(loss=target, metric=target)


class MetricTargetAdapter(TargetAdapter):
    """Pass-through adapter for metric-learning tasks (ranking + contrastive).

    Supervision is implicit in the data structure, not a per-sample label, so the
    ``target`` is passed through as a ``[B]`` float for both loss and metrics
    (kept ``[B]`` — the margin loss needs that shape; unsqueezing would break it).
    It is one of: a dummy ones vector (triplet / InfoNCE / SigLIP — ignored by the
    criterion) or a ``[B]`` ±1 label (pair — consumed by ``MarginRankingCriterion``).
    """

    def adapt(self, target: Tensor) -> TargetView:
        return TargetView(loss=target.float(), metric=target.float())
