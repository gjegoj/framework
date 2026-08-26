"""Contrastive criteria over stacked view embeddings ``[B, N, D]``, supervised in-batch."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from src.losses.base import WrappedCriterion, split_views
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor


def _normalized_pair(logits: Tensor, owner: str) -> tuple[Tensor, Tensor]:
    """The ``[B, 2, D]`` pair as L2-normalized views — what similarity losses consume."""
    first, second = split_views(logits, 2, owner)
    return functional.normalize(first, dim=-1), functional.normalize(second, dim=-1)


class InfoNceLoss(nn.Module):
    """Symmetric InfoNCE over in-batch pairs — the CLIP objective.

    Cross-entropy against the diagonal of the similarity matrix, averaged over both
    directions. The input is ``[B, N, D]`` from a multi-view or multi-stream backbone; the
    ``target`` argument is ignored — supervision *is* the in-batch diagonal.

    Parameters:
        temperature (float): Initial softmax temperature; a learnable log-scale, as in CLIP.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        first, second = _normalized_pair(logits, type(self).__name__)
        similarities = first @ second.T * self.log_scale.exp()
        labels = torch.arange(similarities.shape[0], device=similarities.device)
        return (functional.cross_entropy(similarities, labels) + functional.cross_entropy(similarities.T, labels)) / 2


@criterion_registry.register("infonce")
class InfoNceCriterion(WrappedCriterion):
    """InfoNCE as a criterion.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`InfoNceLoss` (``temperature``).
    """

    part_name: ClassVar[str] = "infonce"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(InfoNceLoss(**kwargs))


class SigLipLoss(nn.Module):
    """Pairwise sigmoid loss — the SigLIP objective.

    Every pair is scored independently (positive on the diagonal, negative elsewhere), so no
    batch-wide softmax is needed. Scale and bias are learnable, initialized as in the paper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        first, second = _normalized_pair(logits, type(self).__name__)
        pairwise = first @ second.T * self.log_scale.exp() + self.bias
        labels = 2 * torch.eye(pairwise.shape[0], device=pairwise.device) - 1
        return -functional.logsigmoid(labels * pairwise).mean()


@criterion_registry.register("siglip")
class SigLipCriterion(WrappedCriterion):
    """SigLIP as a criterion."""

    part_name: ClassVar[str] = "siglip"

    def __init__(self) -> None:
        super().__init__(SigLipLoss())


class TripletLoss(nn.Module):
    """Triplet margin loss over ``[B, 3, D]`` anchor/positive/negative views.

    Views are consumed raw — the margin is defined in the embedding space the backbone
    produces, so no normalization is applied.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.TripletMarginLoss`` (``margin``, ``p``, ``swap``, ...).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._distance = nn.TripletMarginLoss(**kwargs)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        anchor, positive, negative = split_views(logits, 3, type(self).__name__)
        result: Tensor = self._distance(anchor, positive, negative)
        return result


@criterion_registry.register("triplet")
class TripletCriterion(WrappedCriterion):
    """Triplet margin as a criterion.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`TripletLoss`.
    """

    part_name: ClassVar[str] = "triplet"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(TripletLoss(**kwargs))
