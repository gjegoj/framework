"""Classification criteria: cross-entropy, BCE, and focal (multiclass / binary / multilabel).

Wrappers declare only the parameters that need conversion or a default; everything else
forwards verbatim to the wrapped torch loss (see ``SingleTermCriterion``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.losses.base import SingleTermCriterion
from src.losses.registry import criteria


@criteria.register("cross_entropy")
class CrossEntropyCriterion(SingleTermCriterion):
    """Multiclass cross-entropy on logits ``[B, C]`` vs class indices ``[B]``.

    Parameters:
        weight (list[float] | None): Optional per-class rescaling weights.
        **kwargs: Forwarded verbatim to ``nn.CrossEntropyLoss``
            (``label_smoothing``, ``ignore_index``, ``reduction``, ...).
    """

    component_name = "cross_entropy"

    def __init__(
        self,
        weight: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        weight_tensor = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        super().__init__(nn.CrossEntropyLoss(weight=weight_tensor, **kwargs))


@criteria.register("bce")
class BCECriterion(SingleTermCriterion):
    """Binary / multilabel BCE on logits vs float targets.

    Operates on logits ``[B]`` or ``[B, C]`` vs float targets of the same shape.
    Use for binary (out=1) and multilabel (out=C) objectives.

    Parameters:
        pos_weight (list[float] | None): Per-class positive-class weight (length C).
        **kwargs: Forwarded verbatim to ``nn.BCEWithLogitsLoss`` (``reduction``, ...).
    """

    component_name = "bce"

    def __init__(self, pos_weight: list[float] | None = None, **kwargs: Any) -> None:
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
        super().__init__(nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, **kwargs))


class FocalLoss(nn.Module):
    """Multiclass focal loss on logits with optional per-class weighting.

    Computes ``-alpha_c * (1 - p_t) ** gamma * log p_t`` where ``p_t`` is the softmax
    probability assigned to the target class. Works at any spatial rank, so one instance
    serves both classification and segmentation.

    Parameters:
        alpha (list[float] | None): Per-class weights (length ``C``). ``None`` → unweighted.
        gamma (float): Focusing parameter (``>= 0``); higher down-weights easy examples more.
            ``0`` recovers (weighted) cross-entropy.
        reduction (str): ``"mean"`` (default) / ``"sum"`` / ``"none"``.
    """

    alpha: Tensor | None

    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'/'sum'/'none', got {reduction!r}.")
        weights = torch.tensor(alpha, dtype=torch.float) if alpha is not None else None
        # Buffer so the weights follow the module across .to(device) and state_dict.
        self.register_buffer("alpha", weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probability = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probability = target_log_probability.exp()
        loss = -(1.0 - target_probability).pow(self.gamma) * target_log_probability
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@criteria.register("focal")
class FocalCriterion(SingleTermCriterion):
    """Focal loss on logits — down-weights well-classified examples.

    Multiclass, generalised across GLOBAL (``[B, C]`` vs ``[B]``) and DENSE
    (``[B, C, H, W]`` vs ``[B, H, W]``) topologies. With a per-class ``alpha`` it is the
    weighted focal loss. (``smp.losses.FocalLoss`` — scalar alpha, binary/multilabel modes —
    stays reachable via a ``_target_`` spec when those are what you need instead.)

    Parameters:
        alpha (list[float] | None): Per-class weights (length ``C``). ``None`` → unweighted.
        gamma (float): Focusing parameter (default ``2.0``).
        reduction (str): ``"mean"`` (default) / ``"sum"`` / ``"none"``.
    """

    component_name = "focal"

    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__(FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction))
