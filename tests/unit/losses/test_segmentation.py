"""Segmentation criteria: dice, IoU, Tversky."""

from __future__ import annotations

import pytest
import torch

from src.losses import DiceCriterion, IoUCriterion, TverskyCriterion


def test_dice_returns_a_named_loss_on_dense_shapes() -> None:
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))

    loss = DiceCriterion()(logits, target)

    assert set(loss.parts) == {"dice"}
    assert loss.total.requires_grad
    assert loss.total.shape == ()


def test_perfect_prediction_scores_near_zero() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 2, 4, 4), -20.0)
    logits[:, 0] = 20.0  # Overwhelming probability for the correct class.

    loss = DiceCriterion()(logits, target)

    assert loss.total.item() == pytest.approx(0.0, abs=1e-3)


def test_binary_mode_forwards_via_kwargs() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()

    loss = DiceCriterion(mode="binary")(logits, target)

    assert loss.total.shape == ()


@pytest.mark.parametrize(
    ("criterion", "part"),
    [(IoUCriterion, "iou"), (TverskyCriterion, "tversky")],
    ids=["iou", "tversky"],
)
def test_the_smp_family_shares_one_contract(criterion: type, part: str) -> None:
    """Same shapes, own part name — a config swaps them without touching anything else."""
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))

    loss = criterion()(logits, target)

    assert set(loss.parts) == {part}
    assert loss.total.requires_grad
    assert loss.total.shape == ()


def test_tversky_penalties_forward_via_kwargs() -> None:
    """``alpha``/``beta`` are the point of Tversky — they must reach smp untouched."""
    logits = torch.randn(2, 3, 8, 8)

    loss = TverskyCriterion(alpha=0.7, beta=0.3)(logits, torch.randint(0, 3, (2, 8, 8)))

    assert loss.total.shape == ()
