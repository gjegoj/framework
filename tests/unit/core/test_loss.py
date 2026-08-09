"""``Loss`` contract: one class covers single, weighted, and composite losses.

Key decision: there is no separate loss-aggregator entity — the weighted
multi-task total is expressed with plain ``Loss`` arithmetic.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.core import Loss


def scalar(value: float) -> Tensor:
    return torch.tensor(value)


def test_part_builds_a_single_part_loss() -> None:
    loss = Loss.part("ce", scalar(0.7))

    assert loss.total.item() == pytest.approx(0.7)
    assert set(loss.parts) == {"ce"}
    assert loss.parts["ce"].item() == pytest.approx(0.7)


def test_add_sums_totals_and_merges_parts() -> None:
    loss = Loss.part("ce", scalar(1.0)) + Loss.part("dice", scalar(0.5))

    assert loss.total.item() == pytest.approx(1.5)
    assert set(loss.parts) == {"ce", "dice"}


def test_add_rejects_colliding_part_names() -> None:
    with pytest.raises(ValueError, match="ce"):
        _ = Loss.part("ce", scalar(1.0)) + Loss.part("ce", scalar(2.0))


def test_weight_scales_total_and_every_part() -> None:
    loss = 0.5 * (Loss.part("ce", scalar(1.0)) + Loss.part("dice", scalar(0.4)))

    assert loss.total.item() == pytest.approx(0.7)
    assert loss.parts["ce"].item() == pytest.approx(0.5)
    assert loss.parts["dice"].item() == pytest.approx(0.2)


def test_weight_applies_from_either_side() -> None:
    left = 2.0 * Loss.part("ce", scalar(1.0))
    right = Loss.part("ce", scalar(1.0)) * 2.0

    assert left.total.item() == right.total.item() == pytest.approx(2.0)


def test_scoped_prefixes_parts_and_keeps_total() -> None:
    loss = Loss.part("ce", scalar(1.0)).scoped("age")

    assert loss.total.item() == pytest.approx(1.0)
    assert set(loss.parts) == {"age/ce"}


def test_sum_folds_many_losses_into_one() -> None:
    total = Loss.sum(Loss.part(name, scalar(1.0)) for name in ("a", "b", "c"))

    assert total.total.item() == pytest.approx(3.0)
    assert set(total.parts) == {"a", "b", "c"}


def test_sum_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        Loss.sum(())


def test_builtin_sum_works_at_runtime() -> None:
    total = sum([Loss.part("a", scalar(1.0)), Loss.part("b", scalar(2.0))])

    assert isinstance(total, Loss)
    assert total.total.item() == pytest.approx(3.0)


def test_multitask_weighted_total_is_plain_arithmetic() -> None:
    """Multi-task step scenario: total = Σ weight · task loss, no aggregator involved."""
    per_task = {
        "seg": Loss.part("ce", scalar(1.0)) + Loss.part("dice", scalar(0.5)),
        "cls": Loss.part("ce", scalar(2.0)),
    }
    weights = {"seg": 1.0, "cls": 0.5}

    total = Loss.sum(weights[name] * loss.scoped(name) for name, loss in per_task.items())

    assert total.total.item() == pytest.approx(2.5)
    assert set(total.parts) == {"seg/ce", "seg/dice", "cls/ce"}
    assert total.parts["cls/ce"].item() == pytest.approx(1.0)
