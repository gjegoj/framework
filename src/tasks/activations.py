"""Activations: map logits to predictions for metrics/inference (never for loss)."""

from __future__ import annotations

from torch import Tensor

from src.core.ports import Activation


class SoftmaxActivation(Activation):
    """Softmax over the class dimension (multiclass)."""

    def __call__(self, logits: Tensor) -> Tensor:
        return logits.softmax(dim=1)


class SigmoidActivation(Activation):
    """Per-class sigmoid (binary / multilabel)."""

    def __call__(self, logits: Tensor) -> Tensor:
        return logits.sigmoid()


class IdentityActivation(Activation):
    """No-op (regression / continuous targets)."""

    def __call__(self, logits: Tensor) -> Tensor:
        return logits
