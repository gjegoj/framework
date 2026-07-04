"""Tests for the focal loss criterion.

Focal loss is a standard supervised criterion (no new topology/objective). It is
generalised across GLOBAL (``[B, C]`` vs ``[B]``) and DENSE (``[B, C, H, W]`` vs
``[B, H, W]``) shapes, operates on logits, and supports an optional per-class
``alpha`` vector — the distinguishing feature over ``smp.losses.FocalLoss``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from src.losses.focal import FocalCriterion, FocalLoss


class TestRegistration:
    def test_registered(self) -> None:
        import src.losses  # noqa: F401 — importing the package self-registers every criterion
        from src.losses.registry import criteria

        assert "focal" in criteria

    def test_create_from_registry(self) -> None:
        from src.losses.registry import criteria

        criterion = criteria.create("focal", gamma=3.0, alpha=[0.25, 0.75])
        assert isinstance(criterion, FocalCriterion)

    def test_components_key(self) -> None:
        result = FocalCriterion()(torch.randn(3, 5), torch.tensor([0, 1, 2]))
        assert "focal" in result.components
        assert torch.equal(result.components["focal"], result.total)


class TestGlobalShape:
    """Classification: logits ``[B, C]`` vs class-index targets ``[B]``."""

    def test_confident_correct_has_low_loss(self) -> None:
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = torch.tensor([0, 1])
        assert FocalCriterion()(logits, target).total.item() < 0.01

    def test_wrong_prediction_has_higher_loss(self) -> None:
        criterion = FocalCriterion()
        target = torch.tensor([0])
        correct = criterion(torch.tensor([[5.0, 0.0, 0.0]]), target).total
        wrong = criterion(torch.tensor([[0.0, 5.0, 0.0]]), target).total
        assert wrong.item() > correct.item()


# The criterion is generalised over GLOBAL ([B, C] vs [B]) and DENSE ([B, C, H, W] vs [B, H, W])
# shapes — the shape-agnostic behaviors are pinned once for both.
_SHAPE_CASES = [
    pytest.param((8, 4), (8,), id="global"),
    pytest.param((2, 4, 8, 8), (2, 8, 8), id="dense"),
]


class TestShapeGeneralisation:
    @pytest.mark.parametrize(("logits_shape", "target_shape"), _SHAPE_CASES)
    def test_returns_scalar(self, logits_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
        result = FocalCriterion()(torch.randn(*logits_shape), torch.randint(0, 4, target_shape))
        assert result.total.ndim == 0
        assert result.total.item() > 0

    @pytest.mark.parametrize(("logits_shape", "target_shape"), _SHAPE_CASES)
    def test_gamma_zero_matches_cross_entropy(
        self, logits_shape: tuple[int, ...], target_shape: tuple[int, ...]
    ) -> None:
        """With gamma=0 the focal term is 1, so focal reduces to cross-entropy."""
        logits = torch.randn(*logits_shape)
        target = torch.randint(0, 4, target_shape)
        focal = FocalCriterion(gamma=0.0)(logits, target).total
        assert torch.allclose(focal, F.cross_entropy(logits, target), atol=1e-6)


class TestGammaBehaviour:
    def test_gamma_downweights_easy_examples(self) -> None:
        """On confident-correct examples, focusing (gamma>0) lowers the loss vs gamma=0."""
        logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        target = torch.tensor([0, 1, 2])
        no_focus = FocalCriterion(gamma=0.0)(logits, target).total
        focused = FocalCriterion(gamma=2.0)(logits, target).total
        assert focused.item() < no_focus.item()


class TestPerClassAlpha:
    def test_uniform_alpha_scales_loss(self) -> None:
        """A constant per-class alpha=c scales every element, so the mean scales by c."""
        logits = torch.randn(12, 3)
        target = torch.randint(0, 3, (12,))
        unweighted = FocalCriterion()(logits, target).total
        weighted = FocalCriterion(alpha=[2.0, 2.0, 2.0])(logits, target).total
        assert torch.allclose(weighted, 2.0 * unweighted, atol=1e-6)

    def test_alpha_registered_as_buffer(self) -> None:
        """alpha is a buffer so it follows .to(device) and lands in the state_dict."""
        loss = FocalLoss(alpha=[0.5, 1.5])
        assert "alpha" in dict(loss.named_buffers())

    def test_no_alpha_has_no_buffer_value(self) -> None:
        assert FocalLoss(alpha=None).alpha is None


class TestReduction:
    def test_none_keeps_per_element_shape(self) -> None:
        loss = FocalLoss(reduction="none")
        target = torch.randint(0, 4, (3, 6, 6))
        assert loss(torch.randn(3, 4, 6, 6), target).shape == target.shape

    def test_sum_is_mean_times_count(self) -> None:
        logits = torch.randn(10, 3)
        target = torch.randint(0, 3, (10,))
        total_sum = FocalLoss(reduction="sum")(logits, target)
        total_mean = FocalLoss(reduction="mean")(logits, target)
        assert torch.allclose(total_sum, total_mean * 10, atol=1e-5)


class TestValidationAndGradients:
    @pytest.mark.parametrize(
        ("build", "match"),
        [
            pytest.param(lambda: FocalLoss(gamma=-1.0), "gamma must be non-negative", id="negative_gamma"),
            pytest.param(lambda: FocalLoss(reduction="average"), "reduction must be", id="unknown_reduction"),
        ],
    )
    def test_invalid_constructor_argument_raises(self, build: Callable[[], FocalLoss], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build()

    def test_gradients_flow_to_logits(self) -> None:
        logits = torch.randn(8, 4, requires_grad=True)
        FocalCriterion(gamma=2.0, alpha=[1.0, 2.0, 1.0, 0.5])(logits, torch.randint(0, 4, (8,))).total.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
