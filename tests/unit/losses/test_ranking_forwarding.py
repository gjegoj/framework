"""Ranking criteria: verbatim kwargs forwarding to the functional torch losses."""

from __future__ import annotations

import torch

from src.losses.registry import criteria


class TestRankingKwargsForwarding:
    def test_triplet_forwards_swap_and_reduction(self) -> None:
        """Unlisted F.triplet_margin_loss params (swap/eps/reduction) are configurable from YAML."""
        criterion = criteria.create("triplet_margin", margin=0.5, swap=True, reduction="sum")
        logits = torch.randn(4, 3, 8)
        result = criterion(logits, torch.zeros(4))
        assert result.total.ndim == 0  # reduction="sum" still yields a scalar; swap accepted at all

    def test_margin_ranking_forwards_reduction(self) -> None:
        criterion = criteria.create("margin_ranking", reduction="none")
        logits = torch.randn(4, 2, 1)
        result = criterion(logits, torch.ones(4))
        assert result.total.shape == (4,)  # reduction="none" preserved per-pair losses
