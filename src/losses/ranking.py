"""Ranking loss criteria for MULTIVIEW topology.

Each criterion receives ``logits: [B, N, D]`` — N embedding vectors per sample —
and a ``target: [B]`` label tensor (see notes below for each criterion).

The ``[B, N, D]`` contract is enforced via ``base.require_view_shape`` so shape
mismatches surface early rather than inside PyTorch's loss functions. Wrappers
declare only the parameters with framework defaults; everything else forwards
verbatim to the underlying functional loss.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.base import require_view_shape
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
        **kwargs: Forwarded verbatim to ``F.triplet_margin_loss``
            (``p``, ``swap``, ``eps``, ``reduction``, ...).
    """

    def __init__(self, margin: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self._margin = margin
        self._loss_kwargs = kwargs

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        require_view_shape(logits, views=3, owner=type(self).__name__)
        anchor, positive, negative = logits[:, 0], logits[:, 1], logits[:, 2]
        value: Tensor = F.triplet_margin_loss(anchor, positive, negative, margin=self._margin, **self._loss_kwargs)
        return LossResult(total=value, components={"triplet_margin": value})


@criteria.register("margin_ranking")
class MarginRankingCriterion(Criterion):
    """Margin ranking loss on ``[B, 2, D]`` embeddings.

    Scores each view (raw scalar when the head is 1-D, else the embedding's L2 norm — see
    ``_view_score``), then applies ``F.margin_ranking_loss``.  ``target[i] = +1`` means the first
    view should score higher; ``-1`` means the second should score higher.

    Parameters:
        margin (float): Minimum desired score gap between views.
        **kwargs: Forwarded verbatim to ``F.margin_ranking_loss`` (``reduction``, ...).
    """

    def __init__(self, margin: float = 0.0, **kwargs: Any) -> None:
        super().__init__()
        self._margin = margin
        self._loss_kwargs = kwargs

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        require_view_shape(logits, views=2, owner=type(self).__name__)
        first_score: Tensor = _view_score(logits[:, 0])
        second_score: Tensor = _view_score(logits[:, 1])
        value: Tensor = F.margin_ranking_loss(
            first_score, second_score, target, margin=self._margin, **self._loss_kwargs
        )
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

    Parameters:
        **kwargs: Forwarded verbatim to ``F.binary_cross_entropy_with_logits``
            (``reduction``, ...).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._loss_kwargs = kwargs

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        require_view_shape(logits, views=2, owner=type(self).__name__)
        first_score: Tensor = _view_score(logits[:, 0])
        second_score: Tensor = _view_score(logits[:, 1])
        gap: Tensor = first_score - second_score
        value: Tensor = F.binary_cross_entropy_with_logits(gap, target.to(gap.dtype), **self._loss_kwargs)
        return LossResult(total=value, components={"ranknet": value})
