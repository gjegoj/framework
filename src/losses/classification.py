"""Objectives over class scores: one choice, one yes-or-no, or one per label."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from src.core import FEATURE_AXIS, LossOutput, Semantics
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
        semantics: What this task's labels mean, settled by the task rather than declared. It is what
            tells a share of each class apart from a row of independent yes-or-nos, which have the same
            shape and are different questions — see ``forward``.
    """

    alpha: Tensor | None
    """Registered as a buffer, so per-class weights travel with the model."""

    def __init__(
        self, alpha: list[float] | None = None, gamma: float = 2.0, semantics: Semantics | None = None
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Focal gamma discounts easy examples, so it cannot be negative; got {gamma}.")
        self.gamma = gamma
        self.semantics = semantics
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float))

    def _per_position(self, targets: Tensor) -> Tensor | None:
        """The declared class weight spread over a yes-or-no target: one weight on each side of it.

        Without this, ``alpha`` was honoured only where the target names a class, and a run that
        declared it on a binary task got plain focal loss and no word about it.
        """
        if self.alpha is None:
            return None
        if self.alpha.numel() != 2:
            raise ValueError(
                f"A yes-or-no target has two sides, so focal's alpha needs two weights; "
                f"{self.alpha.numel()} were declared."
            )
        return torch.where(targets > 0.5, self.alpha[1], self.alpha[0])

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        """The probability the right answer was given, and the term discounting it where it was already high.

        Which question is being asked cannot be read off the shapes. A share of each class and a row of
        independent yes-or-nos arrive identically shaped — both as wide as the output — so what tells
        them apart is what the task settled its labels mean. Read from the shapes alone, a multiclass
        target left soft by MixUp or CutMix went down the yes-or-no branch and the run descended a set
        of independent binary questions: measured, the same batch read 0.55476 where cross-entropy read
        0.30643, and moving every logit by the same amount — which a distribution over classes cannot
        notice — moved it to 6.66671.
        """
        outputs = aligned(outputs, targets)
        if outputs.ndim == targets.ndim + 1:  # a class axis against an index per position
            given = outputs.softmax(dim=FEATURE_AXIS).gather(FEATURE_AXIS, targets.long().unsqueeze(FEATURE_AXIS))
            given = given.squeeze(FEATURE_AXIS)
            terms = nn.functional.cross_entropy(outputs, targets.long(), weight=self.alpha, reduction="none")
        elif self.semantics is Semantics.MULTICLASS:  # a share of each class per position
            given, terms = self._over_shares(outputs, targets)
        else:  # one score per position: a yes-or-no
            probabilities = outputs.sigmoid()
            given = torch.where(targets > 0.5, probabilities, 1 - probabilities)
            terms = nn.functional.binary_cross_entropy_with_logits(
                outputs, targets.float(), weight=self._per_position(targets), reduction="none"
            )
        return self.reported((terms * (1 - given).pow(self.gamma)).mean())

    def _over_shares(self, outputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """One distribution against another: the share each class was given, against the share it holds.

        Reduces to the branch above exactly — a share naming one class *is* that class's index — which
        is what keeps `loss: focal` one objective whether or not a run declares a batch transform beside
        it. ``alpha`` weighs each class's own share, as it weighs the named class there.
        """
        logarithms = nn.functional.log_softmax(outputs, dim=FEATURE_AXIS)
        shares = targets
        if self.alpha is not None:
            # Along the class axis, whatever sits after it: `[C]` for a whole-sample answer, `[C, 1, 1]`
            # where a dense one carries height and width behind it.
            shares = targets * self.alpha.reshape(-1, *(1,) * (outputs.ndim - FEATURE_AXIS - 1))
        return (targets * logarithms.exp()).sum(FEATURE_AXIS), -(shares * logarithms).sum(FEATURE_AXIS)
