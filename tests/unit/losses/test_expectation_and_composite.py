"""The two criteria a binned regression is judged by, and how they add up."""

from __future__ import annotations

import pytest
import torch

from src.assembly import instantiate
from src.config import ComponentConfig
from src.losses import (
    CrossEntropyCriterion,
    ExpectationCriterion,
    HuberCriterion,
    MeanSquaredErrorCriterion,
    WeightedSumCriterion,
)
from src.losses.registry import criterion_registry

VALUES = [0.0, 1.0, 2.0, 3.0]


def one_hot(index: int) -> torch.Tensor:
    return torch.eye(len(VALUES))[index].unsqueeze(0)


def test_both_sides_are_reduced_to_the_number_they_stand_for() -> None:
    criterion = ExpectationCriterion(VALUES)
    logits = torch.tensor([[0.0, 100.0, 0.0, 0.0]])  # all but certain about class 1 -> 1.0

    loss = criterion(logits, one_hot(3))  # target stands for 3.0

    assert loss.total.item() == pytest.approx(2.0, abs=1e-4)


def test_a_perfect_prediction_costs_nothing() -> None:
    criterion = ExpectationCriterion(VALUES)

    loss = criterion(torch.tensor([[0.0, 100.0, 0.0, 0.0]]), one_hot(1))

    assert loss.total.item() == pytest.approx(0.0, abs=1e-4)


def test_the_penalty_stays_proportional_to_how_far_the_number_is() -> None:
    """Cross-entropy saturates once the mass stops overlapping; this term must not."""
    criterion = ExpectationCriterion(VALUES)
    near = criterion(torch.tensor([[0.0, 100.0, 0.0, 0.0]]), one_hot(2)).total
    far = criterion(torch.tensor([[0.0, 100.0, 0.0, 0.0]]), one_hot(3)).total

    assert far.item() == pytest.approx(2 * near.item(), abs=1e-4)


def test_it_logs_under_its_own_name() -> None:
    loss = ExpectationCriterion(VALUES)(torch.zeros(1, 4), one_hot(0))

    assert set(loss.parts) == {"expectation"}


def test_the_class_values_stay_out_of_the_checkpoint() -> None:
    """They describe the data, not the trained weights; a later run may bin differently."""
    assert "class_values" not in ExpectationCriterion(VALUES).state_dict()


def test_a_head_of_the_wrong_width_is_reported() -> None:
    with pytest.raises(ValueError, match="must agree"):
        ExpectationCriterion(VALUES)(torch.zeros(1, 7), one_hot(0))


def test_the_weighted_sum_adds_its_parts_and_keeps_their_names() -> None:
    criterion = WeightedSumCriterion([(CrossEntropyCriterion(), 1.0), (ExpectationCriterion(VALUES), 0.5)])
    logits = torch.randn(2, 4)
    target = torch.eye(4)[[1, 2]]

    loss = criterion(logits, target)

    assert set(loss.parts) == {"ce", "expectation"}
    assert loss.total.item() == pytest.approx(sum(part.item() for part in loss.parts.values()), abs=1e-5)


def test_a_weight_scales_the_part_that_is_logged_too() -> None:
    """A logged part must be the number that actually entered the total."""
    alone = ExpectationCriterion(VALUES)(torch.zeros(1, 4), one_hot(3)).total.item()
    weighted = WeightedSumCriterion([(ExpectationCriterion(VALUES), 0.5)])(torch.zeros(1, 4), one_hot(3))

    assert weighted.parts["expectation"].item() == pytest.approx(0.5 * alone, abs=1e-5)


def test_parts_sharing_a_name_are_refused_rather_than_merged() -> None:
    criterion = WeightedSumCriterion([(ExpectationCriterion(VALUES), 1.0), (ExpectationCriterion(VALUES), 1.0)])

    with pytest.raises(ValueError, match="collide"):
        criterion(torch.zeros(1, 4), one_hot(0))


def test_a_sum_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        WeightedSumCriterion([])


def test_the_parts_are_submodules_so_their_buffers_follow_the_model() -> None:
    criterion = WeightedSumCriterion([(ExpectationCriterion(VALUES), 1.0)])

    assert any(buffer.numel() == len(VALUES) for buffer in criterion.buffers())


def test_absolute_error_is_the_default_comparison() -> None:
    criterion = ExpectationCriterion(VALUES)

    loss = criterion(torch.tensor([[0.0, 100.0, 0.0, 0.0]]), one_hot(3))

    assert loss.total.item() == pytest.approx(2.0, abs=1e-4)  # |1 - 3|, not (1 - 3)^2


def test_any_torch_loss_can_compare_the_two_numbers_instead() -> None:
    """Two numbers can be compared in more than one way; the slot takes any module."""
    criterion = ExpectationCriterion(VALUES, distance=MeanSquaredErrorCriterion())

    loss = criterion(torch.tensor([[0.0, 100.0, 0.0, 0.0]]), one_hot(3))

    assert loss.total.item() == pytest.approx(4.0, abs=1e-3)  # (1 - 3)^2


def test_a_declared_comparison_reaches_the_criterion_through_config() -> None:
    criterion = instantiate(
        ComponentConfig.model_validate(
            {
                "name": "expectation",
                "distance": {"_target_": "src.losses.HuberCriterion", "delta": 0.5},
            }
        ),
        criterion_registry,
        class_values=VALUES,
    )

    assert isinstance(criterion._distance, HuberCriterion)
    assert criterion._distance._loss.delta == 0.5  # the knob reached torch untouched


def test_arguments_for_two_different_comparisons_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        ExpectationCriterion(VALUES, distance=MeanSquaredErrorCriterion(), reduction="sum")


def test_the_distance_name_folds_into_expectation() -> None:
    """The term's identity in a composite is 'expectation'; the metric inside is its detail."""
    loss = ExpectationCriterion(VALUES, distance=MeanSquaredErrorCriterion())(torch.zeros(1, 4), one_hot(0))

    assert set(loss.parts) == {"expectation"}
