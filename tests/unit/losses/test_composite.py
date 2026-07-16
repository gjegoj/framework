"""WeightedSumCriterion: term access by YAML label (the criterion-schedule dot-path seam)."""

from __future__ import annotations

import pytest

from src.losses import DiceCriterion, FocalCriterion, WeightedSumCriterion


class TestWeightedSumTermAccess:
    def test_terms_indexable_by_label(self) -> None:
        criterion = WeightedSumCriterion(losses={"focal": {"weight": 2.0, "gamma": 2.0}, "dice": 1.0})
        assert isinstance(criterion["focal"], FocalCriterion)
        assert isinstance(criterion["dice"], DiceCriterion)

    def test_unknown_label_raises_listing_terms(self) -> None:
        criterion = WeightedSumCriterion(losses={"focal": 2.0, "dice": 1.0})
        with pytest.raises(KeyError, match="dice"):
            criterion["focall"]

    def test_forward_still_combines(self) -> None:
        import torch

        criterion = WeightedSumCriterion(losses={"focal": {"weight": 2.0, "gamma": 2.0}, "dice": 1.0})
        logits = torch.randn(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))
        result = criterion(logits, target)
        assert result.total.ndim == 0
        assert {"focal", "dice"} <= result.components.keys()
