"""Ranking loss criteria for RANKING topology (M7a).

Each criterion receives ``logits: [B, N, D]`` — N embedding vectors per sample —
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


def _view_score(view: Tensor) -> Tensor:
    """Reduce a view ``[B, D]`` to a scalar ranking score ``[B]``.

    ``D == 1`` → the raw scalar: the head is a learned scalar relevance score ``f(x)`` (sign
    preserved — the canonical pairwise-ranking form).  ``D > 1`` → the embedding's L2 norm
    (magnitude as score), so the same criterion also works on a shared D-dim embedding backbone.
    """
    return view.squeeze(-1) if view.size(-1) == 1 else view.norm(dim=-1)


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

    Scores each view (raw scalar when the head is 1-D, else the embedding's L2 norm — see
    ``_view_score``), then applies ``F.margin_ranking_loss``.  ``target[i] = +1`` means the first
    view should score higher; ``-1`` means the second should score higher.

    Parameters:
        margin (float): Minimum desired score gap between views.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self._margin = margin

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 3 or logits.size(1) != 2:
            raise ValueError(f"MarginRankingCriterion expects logits of shape [B, 2, D], got {tuple(logits.shape)}.")
        first_score: Tensor = _view_score(logits[:, 0])
        second_score: Tensor = _view_score(logits[:, 1])
        value: Tensor = F.margin_ranking_loss(first_score, second_score, target, margin=self._margin)
        return LossResult(total=value, components={"margin_ranking": value})


@criteria.register("ranknet")
class RankNetCriterion(Criterion):
    """RankNet pairwise loss on ``[B, 2, D]`` embeddings (Burges et al., 2005).

    Scores each view (raw scalar when the head is 1-D — the canonical ``f(x)`` relevance score —
    else the embedding's L2 norm; see ``_view_score``) and applies binary cross-entropy to the gap:
    ``P(first ranks higher) = sigmoid(score_first - score_second)``.  ``target`` is that
    probability — ``1`` = first preferred, ``0`` = second preferred, ``0.5`` = tie (soft targets
    allowed).  Unlike ``margin_ranking`` (hinge, ±1 target), this is the smooth logistic form and
    pairs with a 0/1 label.
    """

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 3 or logits.size(1) != 2:
            raise ValueError(f"RankNetCriterion expects logits of shape [B, 2, D], got {tuple(logits.shape)}.")
        first_score: Tensor = _view_score(logits[:, 0])
        second_score: Tensor = _view_score(logits[:, 1])
        gap: Tensor = first_score - second_score
        value: Tensor = F.binary_cross_entropy_with_logits(gap, target.to(gap.dtype))
        return LossResult(total=value, components={"ranknet": value})
