"""Objectives over class scores: one choice, one yes-or-no, or one per label."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from src.core import CLASS_AXIS, LossOutput
from src.losses.base import Loss, TorchLoss, aligned
from src.losses.registry import loss_registry


@loss_registry.register("cross_entropy")
class CrossEntropy(TorchLoss):
    """The choice between classes; takes an index or, after a batch transform, a share of each class."""

    module_type: ClassVar[type[nn.Module]] = nn.CrossEntropyLoss


@loss_registry.register("bce")
class BinaryCrossEntropy(TorchLoss):
    """One independent yes-or-no per output, from raw scores."""

    module_type: ClassVar[type[nn.Module]] = nn.BCEWithLogitsLoss
    squeezes_channel: ClassVar[bool] = True


@loss_registry.register("focal")
class Focal(Loss):
    """Cross-entropy that stops rewarding the examples it already gets right.

    Every term is scaled by ``(1 - p)**gamma``, where ``p`` is the probability given to the right
    class, so a class that dominates the data stops dominating the gradient. ``alpha`` adds the
    usual per-class weight on top.

    Args:
        alpha: Per-class weight, as a config writes it.
        gamma: How sharply an easy example is discounted; 0 is plain cross-entropy.
    """

    alpha: Tensor | None
    """Registered as a buffer, so per-class weights travel with the model."""

    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Focal gamma discounts easy examples, so it cannot be negative; got {gamma}.")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float))

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        outputs = aligned(outputs, targets)
        if outputs.ndim == targets.ndim:  # one score per position: a yes-or-no
            probabilities = outputs.sigmoid()
            given = torch.where(targets > 0.5, probabilities, 1 - probabilities)
            terms = nn.functional.binary_cross_entropy_with_logits(outputs, targets.float(), reduction="none")
        else:
            given = outputs.softmax(dim=CLASS_AXIS).gather(CLASS_AXIS, targets.long().unsqueeze(CLASS_AXIS))
            given = given.squeeze(CLASS_AXIS)
            terms = nn.functional.cross_entropy(outputs, targets.long(), weight=self.alpha, reduction="none")
        return self.reported((terms * (1 - given).pow(self.gamma)).mean())
