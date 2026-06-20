"""Focal loss criterion (down-weights well-classified examples; optional per-class weights).

Multiclass focal loss on logits, generalised across topologies: the same code consumes
``[B, C]`` vs ``[B]`` (GLOBAL classification) and ``[B, C, H, W]`` vs ``[B, H, W]`` (DENSE
segmentation) — the class axis is dim 1, the target carries the remaining dims. Unlike
``smp.losses.FocalLoss`` (scalar ``alpha``, binary/multilabel modes) this accepts a per-class
``alpha`` vector — the "weighted" of the original prototype loss; reach for the smp variant via
a ``_target_`` spec when you need those other modes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.losses.criterion import _SingleTermCriterion
from src.losses.registry import criteria


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
class FocalCriterion(_SingleTermCriterion):
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

    _component_name = "focal"

    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__(FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction))
