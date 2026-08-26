"""Built-in activations: what the standard objectives turn logits into for metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.ports import Activation


def softmax_probabilities(logits: Tensor) -> Tensor:
    """Class probabilities over the class dimension (multiclass default)."""
    return torch.softmax(logits, dim=1)


def squeeze_single_output(logits: Tensor) -> Tensor:
    """Drop a single-channel dimension; wide outputs pass through.

    Covers ``[B, 1]`` and dense ``[B, 1, H, W]`` alike — the channel dim is
    always position 1.
    """
    return logits.squeeze(1) if logits.dim() > 1 and logits.size(1) == 1 else logits


def sigmoid_probabilities(logits: Tensor) -> Tensor:
    """Per-output probabilities (binary and multilabel default)."""
    return torch.sigmoid(squeeze_single_output(logits))


def identity(logits: Tensor) -> Tensor:
    """Predictions are the raw outputs — e.g. the embeddings of metric tasks."""
    return logits


def expectation_over(class_values: Sequence[float]) -> Activation:
    """Build an activation reading a distribution back as one number.

    The inverse of a binned target encoding: the prediction a metric sees is
    ``softmax(logits) · class_values``, so a binned regression reports the value
    it was asked for rather than a vector of bin probabilities.

    Parameters:
        class_values (Sequence[float]): The number each class position stands for.
    """
    values = torch.as_tensor(list(class_values), dtype=torch.float)

    def expectation(logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1) @ values.to(logits.device)

    return expectation
