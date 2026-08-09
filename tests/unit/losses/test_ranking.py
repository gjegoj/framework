"""Ranking criteria: the preferred item must score higher, by hinge or by odds."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from src.losses import MarginRankingCriterion, RankNetCriterion
from src.losses.ranking import RankNetLoss
from src.losses.registry import criterion_registry


def pair(first_score: float, second_score: float) -> torch.Tensor:
    """A scored pair carrier from single-output heads: ``[B, 2, D=1]``."""
    return torch.tensor([[[first_score], [second_score]]])


def test_the_hinge_is_satisfied_by_a_correct_order_with_a_gap() -> None:
    ordered = MarginRankingCriterion(margin=0.5)(pair(2.0, 1.0), torch.tensor([1.0]))
    inverted = MarginRankingCriterion(margin=0.5)(pair(1.0, 2.0), torch.tensor([1.0]))

    assert ordered.total.item() == pytest.approx(0.0)
    assert inverted.total.item() > 0.0


def test_the_target_sign_says_which_view_should_win() -> None:
    """``-1`` flips the preference without swapping the views."""
    prefer_second = MarginRankingCriterion()(pair(1.0, 2.0), torch.tensor([-1.0]))

    assert prefer_second.total.item() == pytest.approx(0.0)


def test_ranknet_is_bce_on_the_score_gap() -> None:
    """The identity that pins the formula to the paper's."""
    loss = RankNetCriterion()(pair(2.0, 1.0), torch.tensor([1.0]))
    expected = functional.binary_cross_entropy_with_logits(torch.tensor([1.0]), torch.tensor([1.0]))

    assert loss.total.item() == pytest.approx(expected.item(), abs=1e-6)


def test_a_tie_is_a_legal_target() -> None:
    """Graded preference is the point of the logistic form; a hinge cannot say 0.5."""
    tied = RankNetCriterion()(pair(1.0, 1.0), torch.tensor([0.5]))

    assert torch.isfinite(tied.total)
    assert tied.total.item() == pytest.approx(
        functional.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([0.5])).item(), abs=1e-6
    )


def test_a_scalar_head_keeps_its_sign() -> None:
    """A norm would fold -3 and 3 together; the canonical relevance score must not."""
    negative_wins = RankNetLoss()(pair(-1.0, -2.0), torch.tensor([1.0]))
    folded_would_lose = RankNetLoss()(pair(-2.0, -1.0), torch.tensor([1.0]))

    assert negative_wins.item() < folded_would_lose.item()


def test_a_wide_embedding_scores_by_its_norm() -> None:
    """The same criterion serves a shared embedding backbone: magnitude as relevance."""
    strong_first = torch.tensor([[[3.0, 4.0], [0.1, 0.1]]])  # norms 5.0 vs ~0.14

    loss = MarginRankingCriterion()(strong_first, torch.tensor([1.0]))

    assert loss.total.item() == pytest.approx(0.0)


def test_a_wrong_carrier_names_the_criterion() -> None:
    with pytest.raises(ValueError, match="RankNetLoss"):
        RankNetCriterion()(torch.randn(2, 3, 4), torch.tensor([1.0, 0.0]))


def test_each_logs_under_its_own_name() -> None:
    assert set(MarginRankingCriterion()(pair(2.0, 1.0), torch.tensor([1.0])).parts) == {"margin_ranking"}
    assert set(RankNetCriterion()(pair(2.0, 1.0), torch.tensor([1.0])).parts) == {"ranknet"}


def test_both_are_reachable_from_config_by_name() -> None:
    assert isinstance(criterion_registry.create("margin_ranking", margin=0.2), MarginRankingCriterion)
    assert isinstance(criterion_registry.create("ranknet"), RankNetCriterion)
