"""Contrastive criteria: InfoNCE and SigLIP over stacked pair embeddings ``[B, 2, D]``."""

from __future__ import annotations

import pytest
import torch

from src.losses import InfoNceCriterion, SigLipCriterion, TripletCriterion


def matched_pairs(batch: int = 4, dim: int = 8) -> torch.Tensor:
    """Identical view embeddings: every pair matches its diagonal partner."""
    torch.manual_seed(0)
    views = torch.nn.functional.normalize(torch.randn(batch, dim), dim=-1)
    return torch.stack([views, views], dim=1)


def test_infonce_rewards_matching_pairs() -> None:
    matched = InfoNceCriterion()(matched_pairs(), torch.empty(0))

    torch.manual_seed(1)
    shuffled = matched_pairs().clone()
    shuffled[:, 1] = shuffled.flip(0)[:, 1]  # Pair each image with the wrong text.
    mismatched = InfoNceCriterion()(shuffled, torch.empty(0))

    assert set(matched.parts) == {"infonce"}
    assert matched.total.item() < mismatched.total.item()


def test_infonce_temperature_is_learnable() -> None:
    criterion = InfoNceCriterion()

    assert sum(1 for _ in criterion.parameters()) == 1


def test_infonce_rejects_non_pair_carriers() -> None:
    with pytest.raises(ValueError, match="B, 2, D"):
        InfoNceCriterion()(torch.randn(4, 8), torch.empty(0))


def test_siglip_runs_on_pair_carriers_and_learns_scale_and_bias() -> None:
    criterion = SigLipCriterion()

    loss = criterion(matched_pairs(), torch.empty(0))

    assert set(loss.parts) == {"siglip"}
    assert loss.total.shape == ()
    assert sum(1 for _ in criterion.parameters()) == 2


def triplets(anchor_matches_positive: bool) -> torch.Tensor:
    """[B, 3, D] carriers: anchor, positive, negative."""
    torch.manual_seed(0)
    anchor = torch.randn(4, 8)
    negative = anchor + 10.0
    positive = anchor.clone() if anchor_matches_positive else negative.clone()
    return torch.stack([anchor, positive, negative], dim=1)


def test_triplet_scores_zero_when_the_positive_is_the_anchor() -> None:
    loss = TripletCriterion()(triplets(anchor_matches_positive=True), torch.empty(0))

    assert set(loss.parts) == {"triplet"}
    assert loss.total.item() == pytest.approx(0.0)


def test_triplet_penalizes_a_positive_at_the_negative() -> None:
    loss = TripletCriterion()(triplets(anchor_matches_positive=False), torch.empty(0))

    assert loss.total.item() > 0


def test_triplet_margin_forwards_via_kwargs() -> None:
    carrier = triplets(anchor_matches_positive=False)

    narrow = TripletCriterion(margin=0.5)(carrier, torch.empty(0))
    wide = TripletCriterion(margin=5.0)(carrier, torch.empty(0))

    assert narrow.total.item() < wide.total.item()


def test_triplet_rejects_non_triplet_carriers() -> None:
    with pytest.raises(ValueError, match="B, 3, D"):
        TripletCriterion()(matched_pairs(), torch.empty(0))
