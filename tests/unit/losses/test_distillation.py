"""Distillation KL: the student follows the teacher's distribution, softened by T."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from src.losses import KLDivergenceCriterion
from src.losses.distillation import KLDivergenceLoss
from src.losses.registry import criterion_registry

STUDENT = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]])
TEACHER = torch.tensor([[1.8, 0.6, 0.2], [0.1, 1.2, 0.9]])


def manual_kl(temperature: float) -> float:
    student = functional.log_softmax(STUDENT / temperature, dim=1)
    teacher = functional.softmax(TEACHER / temperature, dim=1)
    return float(functional.kl_div(student, teacher, reduction="none").sum(1).mean() * temperature**2)


@pytest.mark.parametrize("temperature", [1.0, 4.0])
def test_the_value_is_temperature_scaled_kl(temperature: float) -> None:
    """T² is not decoration: it keeps soft gradients on the scale of hard ones."""
    loss = KLDivergenceLoss(temperature=temperature)(STUDENT, TEACHER)

    assert loss.item() == pytest.approx(manual_kl(temperature), abs=1e-6)


def test_a_matching_student_costs_nothing() -> None:
    loss = KLDivergenceLoss(temperature=2.0)(STUDENT, STUDENT.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gradients_reach_the_student_only() -> None:
    """The teacher is a target, not a participant — it must stay frozen through this loss."""
    student = STUDENT.clone().requires_grad_()
    teacher = TEACHER.clone().requires_grad_()

    KLDivergenceLoss()(student, teacher).backward()

    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_one_module_serves_dense_shapes() -> None:
    """The class dim is dim 1 and the pixels ride along — feature maps distill too."""
    loss = KLDivergenceLoss(temperature=2.0)(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4))

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_an_annealed_temperature_acts_on_the_next_step() -> None:
    """Read per forward, so the `anneal` callback can cool it over the run."""
    criterion = KLDivergenceLoss(temperature=1.0)
    before = criterion(STUDENT, TEACHER)

    criterion.temperature = 4.0
    after = criterion(STUDENT, TEACHER)

    assert after.item() != pytest.approx(before.item())


def test_a_class_label_target_is_refused_with_directions() -> None:
    """Feeding the data-layer label would crash obscurely inside softmax; say what belongs here."""
    with pytest.raises(ValueError, match="teacher"):
        KLDivergenceLoss()(STUDENT, torch.tensor([0, 1]))


def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="temperature"):
        KLDivergenceLoss(temperature=0.0)


def test_it_logs_under_its_own_name() -> None:
    assert set(KLDivergenceCriterion()(STUDENT, TEACHER).parts) == {"kl"}


def test_it_is_reachable_from_config_by_name() -> None:
    assert isinstance(criterion_registry.create("kl_divergence", temperature=4.0), KLDivergenceCriterion)
