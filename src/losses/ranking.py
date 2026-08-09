"""Ranking criteria: which of two items should come first."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn
from torch.nn import functional

from src.losses.base import WrappedCriterion, split_views
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor


def _score_of(view: Tensor) -> Tensor:
    """One number per sample: the raw scalar of a 1-wide head, else the norm.

    A single-output head is the canonical learned relevance ``f(x)``, and its sign is
    preserved; a wider embedding is scored by its L2 norm, so the same criteria serve a
    shared embedding backbone without a second code path.
    """
    return view.squeeze(-1) if view.size(-1) == 1 else view.norm(dim=-1)


class PairRankingLoss(nn.Module):
    """torch's margin ranking over a scored pair carrier ``[B, 2, D]``.

    ``target[i] = +1`` says the first view should score higher, ``-1`` the
    second; the hinge fires while the gap is below ``margin``.

    The hinge form — a hard gap or nothing. Where the preference is graded rather than
    binary, :class:`RankNetLoss` is the smooth counterpart.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.MarginRankingLoss``
            (``margin``, ``reduction``, ...).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._hinge = nn.MarginRankingLoss(**kwargs)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        first, second = split_views(logits, 2, type(self).__name__)
        score = _score_of(first)
        result: Tensor = self._hinge(score, _score_of(second), target.to(score.dtype))
        return result


@criterion_registry.register("margin_ranking")
class MarginRankingCriterion(WrappedCriterion):
    """Margin ranking as a criterion.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`PairRankingLoss`.
    """

    part_name: ClassVar[str] = "margin_ranking"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(PairRankingLoss(**kwargs))


class RankNetLoss(nn.Module):
    """Binary cross-entropy on the score gap (Burges et al., 2005).

    ``P(first ranks higher) = sigmoid(score_first - score_second)``, judged
    against a target probability: ``1`` first preferred, ``0`` second, ``0.5``
    a tie — graded preferences are the point of the logistic form.

    Parameters:
        **kwargs: Forwarded verbatim to
            ``functional.binary_cross_entropy_with_logits`` (``reduction``, ...).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._options = kwargs

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        first, second = split_views(logits, 2, type(self).__name__)
        gap = _score_of(first) - _score_of(second)
        return functional.binary_cross_entropy_with_logits(gap, target.to(gap.dtype), **self._options)


@criterion_registry.register("ranknet")
class RankNetCriterion(WrappedCriterion):
    """RankNet as a criterion.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`RankNetLoss`.
    """

    part_name: ClassVar[str] = "ranknet"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(RankNetLoss(**kwargs))
