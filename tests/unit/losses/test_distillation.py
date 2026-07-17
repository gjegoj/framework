"""KL-divergence distillation criterion: temperature scaling, shape generalisation, gradients."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.losses.distillation import KLDivergenceCriterion


def manual_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """Independent reference: KL(softmax(t/T) || softmax(s/T)) * T^2, mean over batch/pixels."""
    teacher_probabilities = F.softmax(teacher / temperature, dim=1)
    log_ratio = F.log_softmax(teacher / temperature, dim=1) - F.log_softmax(student / temperature, dim=1)
    return (teacher_probabilities * log_ratio).sum(dim=1).mean() * temperature**2


class TestRegistration:
    def test_registered(self) -> None:
        import src.losses  # noqa: F401 — importing the package self-registers every criterion
        from src.losses.registry import criteria

        assert "kl_divergence" in criteria

    def test_create_from_registry(self) -> None:
        from src.losses.registry import criteria

        criterion = criteria.create("kl_divergence", temperature=4.0)
        assert isinstance(criterion, KLDivergenceCriterion)
        assert criterion.temperature == 4.0


class TestValue:
    def test_identical_logits_give_zero(self) -> None:
        logits = torch.randn(4, 3)
        assert KLDivergenceCriterion()(logits, logits.clone()).total.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("temperature", [1.0, 2.0, 4.0])
    def test_matches_manual_formula(self, temperature: float) -> None:
        student, teacher = torch.randn(6, 4), torch.randn(6, 4)
        result = KLDivergenceCriterion(temperature=temperature)(student, teacher)
        assert result.total.item() == pytest.approx(manual_kl(student, teacher, temperature).item(), abs=1e-5)
        assert torch.equal(result.components["kl"], result.total)

    def test_dense_shapes_supported(self) -> None:
        student, teacher = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        result = KLDivergenceCriterion(temperature=2.0)(student, teacher)
        assert result.total.ndim == 0
        assert result.total.item() == pytest.approx(manual_kl(student, teacher, 2.0).item(), abs=1e-5)


class TestGradients:
    def test_gradients_reach_student_only(self) -> None:
        student = torch.randn(4, 3, requires_grad=True)
        teacher = torch.randn(4, 3, requires_grad=True)
        KLDivergenceCriterion()(student, teacher).total.backward()
        assert student.grad is not None and torch.isfinite(student.grad).all()
        assert teacher.grad is None


class TestValidation:
    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_non_positive_temperature_raises(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            KLDivergenceCriterion(temperature=temperature)
