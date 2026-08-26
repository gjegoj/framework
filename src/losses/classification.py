"""Classification criteria: cross-entropy, its binary/multilabel form, and focal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import torch
from torch import nn
from torch.nn import functional

from src.core.choices import one_of
from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor

type Reduction = Literal["mean", "sum", "none"]
"""How a per-sample loss becomes the number that is back-propagated."""


@criterion_registry.register("cross_entropy")
class CrossEntropyCriterion(WrappedCriterion):
    """Multiclass cross-entropy on logits ``[B, C]`` vs class indices ``[B]``.

    Parameters:
        weight (list[float] | None): Per-class rescaling — a plain list so a
            config file can express it; converted to a tensor here.
        **kwargs: Forwarded verbatim to ``nn.CrossEntropyLoss``
            (``label_smoothing``, ``ignore_index``, ...).
    """

    part_name: ClassVar[str] = "ce"

    def __init__(self, weight: list[float] | None = None, **kwargs: Any) -> None:
        weight_tensor = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        super().__init__(nn.CrossEntropyLoss(weight=weight_tensor, **kwargs))


@criterion_registry.register("bce")
class BinaryCrossEntropyCriterion(WrappedCriterion):
    """Binary and multilabel cross-entropy on raw logits.

    Accepts a single-output head against channel-free targets — ``[B, 1]``
    logits vs ``[B]``, or dense ``[B, 1, H, W]`` vs ``[B, H, W]`` — by
    squeezing the channel dimension (preventing a silent broadcast), as well
    as multilabel shapes (``[B, C]`` both).

    Parameters:
        pos_weight (list[float] | None): Positive-class weighting — a plain
            list so a config file can express it; converted to a tensor here.
        **kwargs: Forwarded verbatim to ``nn.BCEWithLogitsLoss``.
    """

    part_name: ClassVar[str] = "bce"

    def __init__(self, pos_weight: list[float] | None = None, **kwargs: Any) -> None:
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
        super().__init__(nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, **kwargs))

    @override
    def _prepare(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if logits.dim() == target.dim() + 1:
            logits = logits.squeeze(1)  # The channel dim: [B, 1] and [B, 1, H, W] alike.
        return logits, target


class FocalLoss(nn.Module):
    """Multiclass focal loss on logits: cross-entropy that fades on easy examples.

    ``-alpha_t * (1 - p_t) ** gamma * log p_t``. The class dimension is dim 1 and the rest
    is carried along, so one module serves ``[B, C]`` and ``[B, C, H, W]`` alike. A soft
    target (a mixed sample's class shares) weights ``log p`` and ``alpha``; for a one-hot
    target that is exactly the hard formula, so the two need no flag, only their dtype.

    Parameters:
        alpha (list[float] | None): Per-class weights, length C. ``None`` keeps classes equal.
        gamma (float): How hard easy examples fade; 0 recovers cross-entropy.
        reduction (str): ``mean``, ``sum`` or ``none``.
        eps (float): Floor for ``1 - p_t``: at ``p_t == 1.0`` in fp32, ``pow``'s backward is
            infinite for ``gamma < 1``, a domain an annealed gamma crosses.
    """

    alpha: Tensor | None

    def __init__(
        self, alpha: list[float] | None = None, gamma: float = 2.0, reduction: Reduction = "mean", eps: float = 1e-6
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"FocalLoss gamma must be non-negative, got {gamma}.")
        reduction = one_of(reduction, Reduction)
        if eps <= 0:
            raise ValueError(f"FocalLoss eps must be positive, got {eps}.")
        weights = torch.tensor(alpha, dtype=torch.float) if alpha is not None else None
        # Not persisted: like expectation's class values, alpha describes the recipe,
        # not the trained weights — a later run may weigh its classes differently.
        self.register_buffer("alpha", weights, persistent=False)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if self.alpha is not None and self.alpha.numel() != logits.shape[1]:
            raise ValueError(
                f"FocalLoss has {self.alpha.numel()} alpha weight(s) but the head produced "
                f"{logits.shape[1]} classes; declare one alpha per class."
            )
        log_probabilities = functional.log_softmax(logits, dim=1)
        if target.is_floating_point():
            target_log_probability = (log_probabilities * target).sum(dim=1)
            spread = self.alpha.view(1, -1, *([1] * (target.dim() - 2))) if self.alpha is not None else None
            weight = (spread * target).sum(dim=1) if spread is not None else None
        else:
            target_log_probability = log_probabilities.gather(1, target.long().unsqueeze(1)).squeeze(1)
            weight = self.alpha[target] if self.alpha is not None else None
        probability = target_log_probability.exp()
        base = (1.0 - probability).clamp_min(self.eps)
        loss = -base.pow(self.gamma) * target_log_probability
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@criterion_registry.register("focal")
class FocalCriterion(WrappedCriterion):
    """Focal loss as a criterion — see :class:`FocalLoss` for the math and every knob.

    ``gamma`` is a plain number, so the ``anneal`` callback can move it over the run. For
    the binary and multilabel forms, reach ``smp.losses.FocalLoss`` by import path.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`FocalLoss` (``alpha``, ``gamma``, ``reduction``, ``eps``).
    """

    part_name: ClassVar[str] = "focal"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(FocalLoss(**kwargs))
