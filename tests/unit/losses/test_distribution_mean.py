"""DistributionMeanCriterion: expectation regression over a binned label distribution."""

from __future__ import annotations

import pytest
import torch

from src.losses.registry import criteria
from src.losses.regression import DistributionMeanCriterion

BIN_EDGES = [0.0, 0.5, 1.0]  # centers: 0.25, 0.75


def _delta_distribution(bin_index: int, num_bins: int = 2) -> torch.Tensor:
    distribution = torch.zeros(1, num_bins)
    distribution[0, bin_index] = 1.0
    return distribution


class TestConstruction:
    def test_registered(self) -> None:
        assert criteria.get("distribution_mean") is DistributionMeanCriterion

    def test_bin_edges_derive_midpoint_centers(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES)
        assert torch.allclose(criterion.bin_centers, torch.tensor([0.25, 0.75]))

    def test_explicit_centers_used_as_is(self) -> None:
        criterion = DistributionMeanCriterion(bin_centers=[0.1, 0.9])
        assert torch.allclose(criterion.bin_centers, torch.tensor([0.1, 0.9]))

    def test_neither_or_both_center_sources_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            DistributionMeanCriterion()
        with pytest.raises(ValueError, match="exactly one"):
            DistributionMeanCriterion(bin_centers=[0.25, 0.75], bin_edges=BIN_EDGES)

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            DistributionMeanCriterion(bin_edges=BIN_EDGES, kind="hinge")


class TestForward:
    def test_matching_expectations_give_zero_loss(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES)
        logits = torch.tensor([[100.0, -100.0]])  # softmax -> delta at bin 0
        result = criterion(logits, _delta_distribution(0))
        assert result.total == pytest.approx(0.0, abs=1e-6)

    def test_l1_distance_between_expectations(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES, kind="l1")
        logits = torch.tensor([[-100.0, 100.0]])  # prediction at center 0.75
        result = criterion(logits, _delta_distribution(0))  # target at center 0.25
        assert result.total == pytest.approx(0.5)
        assert set(result.components) == {"distribution_mean"}

    def test_huber_kind_forwards_delta(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES, kind="huber", delta=1.0)
        logits = torch.tensor([[-100.0, 100.0]])
        result = criterion(logits, _delta_distribution(0))
        assert result.total == pytest.approx(0.5 * 0.5**2)  # quadratic zone: 0.5 * error^2

    def test_gradient_flows_to_logits(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES)
        logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
        criterion(logits, _delta_distribution(0)).total.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_center_count_mismatch_raises(self) -> None:
        criterion = DistributionMeanCriterion(bin_edges=BIN_EDGES)
        with pytest.raises(ValueError, match="2 bin centers"):
            criterion(torch.zeros(1, 3), torch.full((1, 3), 1.0 / 3))


class TestWeightedSumComposition:
    def test_soft_ce_plus_distribution_mean(self) -> None:
        """The LDL recipe: both terms share the same (logits, soft-target) pair."""
        criterion = criteria.create(
            "weighted_sum",
            losses={
                "cross_entropy": 1.0,
                "distribution_mean": {"weight": 0.5, "bin_edges": BIN_EDGES},
            },
        )
        logits = torch.randn(4, 2, requires_grad=True)
        target = torch.softmax(torch.randn(4, 2), dim=1)  # soft label distribution
        result = criterion(logits, target)
        assert set(result.components) == {"cross_entropy", "distribution_mean"}
        assert torch.isfinite(result.total)
        result.total.backward()
        assert logits.grad is not None
