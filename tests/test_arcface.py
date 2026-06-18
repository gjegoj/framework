"""Tests for ArcFace metric learning: cosine head + angular-margin loss.

ArcFace is classification (GLOBAL × multiclass) with two brick swaps — a cosine
classifier head and an additive angular-margin loss — and needs NO new topology
or objective.  These tests pin that decomposition.

TDD order: RED → GREEN → REFACTOR.
"""

from __future__ import annotations

import pytest
import torch

from src.core.keys import IMAGE

# ---------------------------------------------------------------------------
# Cosine classifier head (holds the learnable class prototypes)
# ---------------------------------------------------------------------------


class TestCosineHead:
    def test_registered(self) -> None:
        import src.models.heads  # noqa: F401 — import registers the head builder
        from src.models.registry import head_builders

        assert "cosine" in head_builders

    def test_output_shape_and_cosine_bounded(self) -> None:
        from src.models.registry import head_builders

        head = head_builders.create("cosine", in_features=16, out_features=10)
        out = head(torch.randn(4, 16))
        assert out.shape == (4, 10)
        assert out.min() >= -1.0001 and out.max() <= 1.0001  # cosine ∈ [-1, 1]

    def test_embedding_bottleneck(self) -> None:
        from src.models.registry import head_builders

        head = head_builders.create("cosine", in_features=16, out_features=10, embedding_dim=8)
        assert head(torch.randn(4, 16)).shape == (4, 10)

    def test_prototypes_are_learnable(self) -> None:
        from src.models.registry import head_builders

        head = head_builders.create("cosine", in_features=16, out_features=10)
        assert any(p.requires_grad for p in head.parameters())


# ---------------------------------------------------------------------------
# ArcFace criterion (stateless additive angular margin)
# ---------------------------------------------------------------------------


class TestArcFaceCriterion:
    def test_registered(self) -> None:
        import src.losses.angular  # noqa: F401 — import registers the criterion
        from src.losses.registry import criteria

        assert "arcface" in criteria

    def test_confident_correct_has_low_loss(self) -> None:
        from src.losses.angular import ArcFaceCriterion

        crit = ArcFaceCriterion(margin=0.5, scale=64.0)
        cosine = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        target = torch.tensor([0, 1])
        assert crit(cosine, target).total.item() < 0.05

    def test_wrong_prediction_has_higher_loss(self) -> None:
        from src.losses.angular import ArcFaceCriterion

        crit = ArcFaceCriterion()
        target = torch.tensor([0])
        correct = crit(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), target).total
        wrong = crit(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), target).total
        assert wrong.item() > correct.item()

    def test_margin_increases_loss(self) -> None:
        """The angular margin penalizes the target logit → higher loss than no margin."""
        from src.losses.angular import ArcFaceCriterion

        cosine = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
        target = torch.tensor([0])
        no_margin = ArcFaceCriterion(margin=0.0)(cosine, target).total
        with_margin = ArcFaceCriterion(margin=0.5)(cosine, target).total
        assert with_margin.item() > no_margin.item()

    def test_is_stateless(self) -> None:
        """Margin/scale are fixed — the only trained params are the head prototypes."""
        from src.losses.angular import ArcFaceCriterion

        assert list(ArcFaceCriterion().parameters()) == []

    def test_components_key(self) -> None:
        from src.losses.angular import ArcFaceCriterion

        result = ArcFaceCriterion()(torch.rand(3, 5) * 2 - 1, torch.tensor([0, 1, 2]))
        assert "arcface" in result.components

    def test_wrong_ndim_raises(self) -> None:
        from src.losses.angular import ArcFaceCriterion

        with pytest.raises(ValueError, match=r"\[B, C\]"):
            ArcFaceCriterion()(torch.randn(4, 3, 8), torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Composition: ArcFace = classification preset + cosine head + arcface loss
# ---------------------------------------------------------------------------


class TestArcFaceComposition:
    def test_built_via_classification_preset(self) -> None:
        """No new topology/objective — just a head + loss swap on classification."""
        from src.losses.angular import ArcFaceCriterion
        from src.tasks.presets import classification

        task = classification(
            "identity",
            num_classes=10,
            head={"kind": "cosine", "embedding_dim": 8},
            loss={"name": "arcface", "margin": 0.5, "scale": 64},
        )
        assert task.head_spec.kind == "cosine"
        assert isinstance(task.criterion, ArcFaceCriterion)

    def test_end_to_end_forward_and_loss(self) -> None:
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks.presets import classification

        task = classification(
            "identity",
            num_classes=10,
            head={"kind": "cosine", "embedding_dim": 8},
            loss={"name": "arcface"},
        )
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"identity": task.head_spec})
        logits = model({IMAGE: torch.randn(4, 16)}).task_logits["identity"]
        assert logits.shape == (4, 10)
        loss = task.criterion(logits, torch.randint(0, 10, (4,)))
        assert loss.total.item() > 0
