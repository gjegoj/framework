"""The criteria registry: built-in losses are reachable by config-friendly names."""

from __future__ import annotations

import torch

from src.losses import CrossEntropyCriterion
from src.losses.registry import criterion_registry


def test_built_in_criteria_are_registered() -> None:
    assert set(criterion_registry) == {
        "arcface",
        "arcface_proxy",
        "cross_entropy",
        "expectation",
        "bce",
        "focal",
        "mse",
        "mae",
        "huber",
        "smooth_l1",
        "dice",
        "iou",
        "tversky",
        "infonce",
        "kl_divergence",
        "margin_ranking",
        "ranknet",
        "siglip",
        "triplet",
    }


def test_create_forwards_kwargs_to_the_wrapper() -> None:
    criterion = criterion_registry.create("cross_entropy", label_smoothing=0.1)

    assert isinstance(criterion, CrossEntropyCriterion)
    loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
    assert set(loss.parts) == {"ce"}
