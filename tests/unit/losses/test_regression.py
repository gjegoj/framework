"""Regression criteria: mean squared error."""

from __future__ import annotations

import pytest
import torch

from src.losses import (
    HuberCriterion,
    MeanAbsoluteErrorCriterion,
    MeanSquaredErrorCriterion,
    SmoothL1Criterion,
)


def test_mse_returns_a_named_loss() -> None:
    logits = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([1.0, 4.0])

    loss = MeanSquaredErrorCriterion()(logits, target)

    assert set(loss.parts) == {"mse"}
    assert loss.total.item() == pytest.approx(2.0)


def test_mse_squeezes_the_channel_on_dense_shapes() -> None:
    logits = torch.zeros(2, 1, 4, 4)
    target = torch.ones(2, 4, 4)

    loss = MeanSquaredErrorCriterion()(logits, target)

    assert loss.total.item() == pytest.approx(1.0)


def test_mse_accepts_matching_shapes() -> None:
    logits = torch.zeros(3)
    target = torch.ones(3)

    loss = MeanSquaredErrorCriterion()(logits, target)

    assert loss.total.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("criterion", "part"),
    [(MeanAbsoluteErrorCriterion, "mae"), (HuberCriterion, "huber"), (SmoothL1Criterion, "smooth_l1")],
    ids=["mae", "huber", "smooth_l1"],
)
def test_the_regression_family_shares_one_contract(criterion: type, part: str) -> None:
    """Same shapes, own part name — a config swaps them without touching anything else."""
    outputs = torch.randn(4, 1, requires_grad=True)

    loss = criterion()(outputs, torch.randn(4))

    assert set(loss.parts) == {part}
    assert loss.total.requires_grad
    assert loss.total.shape == ()


def test_huber_and_smooth_l1_agree_only_at_the_default_knobs() -> None:
    """They are two scalings of one shape; both stay because papers cite both names."""
    outputs, target = torch.tensor([0.0, 3.0]), torch.tensor([1.0, 0.0])

    huber = HuberCriterion(delta=1.0)(outputs, target).total
    smooth = SmoothL1Criterion(beta=1.0)(outputs, target).total
    scaled = SmoothL1Criterion(beta=2.0)(outputs, target).total

    assert huber.item() == pytest.approx(smooth.item())
    assert scaled.item() != pytest.approx(smooth.item())
