"""Ranking loss criteria for RANKING topology (M7a).

Both criteria receive ``logits: [B, N, D]`` — N embedding vectors per sample —
and a ``target: [B]`` label tensor (see notes below for each criterion).

The ``[B, N, D]`` contract is enforced explicitly so shape mismatches surface
early rather than inside PyTorch's loss functions.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.registry import criteria


@criteria.register("triplet_margin")
class TripletMarginCriterion(Criterion):
    """Triplet margin loss on ``[B, 3, D]`` embeddings.

    Expects ``logits[:, 0]`` = anchor, ``[:, 1]`` = positive, ``[:, 2]`` =
    negative.  ``target`` is ignored (triplet supervision is implicit in the
    view ordering).

    Parameters:
        margin (float): Minimum desired gap between d(a,p) and d(a,n).
        p (float): Norm degree for distance computation.
    """

    def __init__(self, margin: float = 1.0, p: float = 2.0) -> None:
        super().__init__()
        self._margin = margin
        self._p = p

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 3 or logits.size(1) != 3:
            raise ValueError(f"TripletMarginCriterion expects logits of shape [B, 3, D], got {tuple(logits.shape)}.")
        anchor, positive, negative = logits[:, 0], logits[:, 1], logits[:, 2]
        value: Tensor = F.triplet_margin_loss(anchor, positive, negative, margin=self._margin, p=self._p)
        return LossResult(total=value, components={"triplet": value})


@criteria.register("margin_ranking")
class MarginRankingCriterion(Criterion):
    """Margin ranking loss on ``[B, 2, D]`` embeddings.

    Scores each view by its L2 norm (a simple scalar proxy for relevance), then
    applies ``F.margin_ranking_loss``.  ``target[i] = +1`` means the first view
    should score higher; ``-1`` means the second should score higher.

    Parameters:
        margin (float): Minimum desired score gap between views.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self._margin = margin

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 3 or logits.size(1) != 2:
            raise ValueError(f"MarginRankingCriterion expects logits of shape [B, 2, D], got {tuple(logits.shape)}.")
        # Use L2 norm as a scalar score for each embedding.
        score1: Tensor = logits[:, 0].norm(dim=-1)
        score2: Tensor = logits[:, 1].norm(dim=-1)
        value: Tensor = F.margin_ranking_loss(score1, score2, target, margin=self._margin)
        return LossResult(total=value, components={"margin_ranking": value})
