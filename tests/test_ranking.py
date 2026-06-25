"""Tests for M7a: shared-encoder ranking (RANKING topology, Siamese).

TDD order: RED → GREEN → REFACTOR.
All classes/functions imported here must exist after the GREEN phase.
"""

from __future__ import annotations

import pytest
import torch

from src.core.keys import IMAGE, POOLED
from src.models.assembly import CompositeModel

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


class TestRankingTopology:
    def test_ranking_topology_registered(self) -> None:
        from src.tasks import Topology, topology_strategies

        assert Topology.RANKING in topology_strategies

    def test_ranking_topology_head_spec_with_explicit_view_keys(self) -> None:
        from src.tasks.strategies.topology import RankingTopology

        topo = RankingTopology(view_keys=("anchor", "positive", "negative"))
        spec = topo.head_spec(out_features=64)
        assert spec.kind == "linear"
        assert spec.out_features == 64
        assert spec.feature_key == POOLED
        assert spec.view_keys == ("anchor", "positive", "negative")

    def test_ranking_topology_default_has_no_view_keys(self) -> None:
        """view_keys=None by default — wiring derives them from data.inputs."""
        from src.tasks.strategies.topology import RankingTopology

        topo = RankingTopology()
        assert topo.view_keys is None
        assert topo.head_spec(out_features=32).view_keys is None


# ---------------------------------------------------------------------------
# Multi-view forward pass
# ---------------------------------------------------------------------------


class TestMultiViewForward:
    def _triplet_model(self, emb_dim: int = 32) -> "CompositeModel":
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks.strategies.topology import RankingTopology

        backbone = EmbeddingBackbone(embedding_dim=emb_dim)
        topo = RankingTopology(view_keys=("anchor", "positive", "negative"))
        specs = {"sim": topo.head_spec(out_features=emb_dim)}
        return build_composite_model(backbone, specs)

    def _pair_model(self, emb_dim: int = 32) -> "CompositeModel":
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks.strategies.topology import RankingTopology

        backbone = EmbeddingBackbone(embedding_dim=emb_dim)
        topo = RankingTopology(view_keys=("query", "doc"))
        specs = {"sim": topo.head_spec(out_features=emb_dim)}
        return build_composite_model(backbone, specs)

    def test_triplet_forward_output_shape(self) -> None:
        B, D = 4, 32
        model = self._triplet_model(D)
        inputs = {k: torch.randn(B, D) for k in ("anchor", "positive", "negative")}
        out = model(inputs)
        assert out.task_logits["sim"].shape == (B, 3, D)

    def test_pair_forward_output_shape(self) -> None:
        B, D = 4, 32
        model = self._pair_model(D)
        inputs = {k: torch.randn(B, D) for k in ("query", "doc")}
        out = model(inputs)
        assert out.task_logits["sim"].shape == (B, 2, D)

    def test_shared_weights_produce_same_output_for_identical_views(self) -> None:
        """Same input on anchor and positive → same embedding (shared backbone)."""
        B, D = 4, 32
        model = self._triplet_model(D)
        x = torch.randn(B, D)
        inputs = {"anchor": x, "positive": x, "negative": torch.randn(B, D)}
        out = model(inputs)
        embeddings = out.task_logits["sim"]
        assert torch.allclose(embeddings[:, 0], embeddings[:, 1], atol=1e-5)

    def test_single_view_path_unchanged(self) -> None:
        """Non-ranking tasks still work exactly as before."""
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks import classification

        backbone = EmbeddingBackbone(embedding_dim=32)
        task = classification("label", num_classes=3)
        model = build_composite_model(backbone, {"label": task.head_spec})
        out = model({IMAGE: torch.randn(4, 32)})
        assert out.task_logits["label"].shape == (4, 3)


# ---------------------------------------------------------------------------
# Loss criteria
# ---------------------------------------------------------------------------


class TestTripletMarginCriterion:
    def test_satisfied_triplet_has_near_zero_loss(self) -> None:
        from src.losses.ranking import TripletMarginCriterion

        # margin must be > 0 per PyTorch; use 1.0 — still zero when d(a,n)-d(a,p) > 1
        crit = TripletMarginCriterion(margin=1.0)
        B, D = 4, 16
        anchor = torch.zeros(B, D)
        positive = torch.zeros(B, D)  # identical → d(a,p) = 0
        negative = torch.ones(B, D) * 10.0  # far → d(a,n) ≈ 40 >> 0 + margin
        logits = torch.stack([anchor, positive, negative], dim=1)  # [B, 3, D]
        result = crit(logits, torch.ones(B))
        assert result.total.item() < 1e-3

    def test_violated_triplet_has_positive_loss(self) -> None:
        from src.losses.ranking import TripletMarginCriterion

        crit = TripletMarginCriterion(margin=1.0)
        B, D = 4, 16
        anchor = torch.zeros(B, D)
        positive = torch.ones(B, D) * 10.0  # far
        negative = torch.zeros(B, D)  # identical to anchor
        logits = torch.stack([anchor, positive, negative], dim=1)
        result = crit(logits, torch.ones(B))
        assert result.total.item() > 0.0

    def test_loss_components_key(self) -> None:
        from src.losses.ranking import TripletMarginCriterion

        crit = TripletMarginCriterion()
        logits = torch.randn(2, 3, 8)
        result = crit(logits, torch.ones(2))
        assert "triplet" in result.components

    def test_wrong_n_views_raises(self) -> None:
        from src.losses.ranking import TripletMarginCriterion

        crit = TripletMarginCriterion()
        with pytest.raises(ValueError, match="3"):
            crit(torch.randn(4, 2, 8), torch.ones(4))  # n_views=2, not 3


class TestMarginRankingCriterion:
    def test_correct_order_low_loss(self) -> None:
        """d(a, query) > d(a, doc) and target=+1 → loss ≈ 0."""
        from src.losses.ranking import MarginRankingCriterion

        crit = MarginRankingCriterion(margin=0.0)
        B, D = 4, 8
        high = torch.ones(B, D) * 5.0
        low = torch.zeros(B, D)
        logits = torch.stack([high, low], dim=1)  # [B, 2, D]
        target = torch.ones(B)  # +1: first ranks higher
        result = crit(logits, target)
        assert result.total.item() < 1e-3

    def test_wrong_order_nonzero_loss(self) -> None:
        from src.losses.ranking import MarginRankingCriterion

        crit = MarginRankingCriterion(margin=0.0)
        B, D = 4, 8
        high = torch.ones(B, D) * 5.0
        low = torch.zeros(B, D)
        logits = torch.stack([high, low], dim=1)
        target = -torch.ones(B)  # -1: second should rank higher (but it's lower)
        result = crit(logits, target)
        assert result.total.item() > 0.0

    def test_loss_components_key(self) -> None:
        from src.losses.ranking import MarginRankingCriterion

        crit = MarginRankingCriterion()
        logits = torch.randn(2, 2, 8)
        result = crit(logits, torch.ones(2))
        assert "margin_ranking" in result.components

    def test_wrong_n_views_raises(self) -> None:
        from src.losses.ranking import MarginRankingCriterion

        crit = MarginRankingCriterion()
        with pytest.raises(ValueError, match="2"):
            crit(torch.randn(4, 3, 8), torch.ones(4))  # n_views=3, not 2

    def test_scalar_head_uses_raw_signed_score(self) -> None:
        """D=1: the raw signed scalar is the score, not its magnitude (norm would flip the order)."""
        from src.losses.ranking import MarginRankingCriterion

        crit = MarginRankingCriterion(margin=0.0)
        logits = torch.zeros(4, 2, 1)
        logits[:, 0, 0] = -5.0  # first view's scalar score is negative → it ranks lower than 0
        result = crit(logits, -torch.ones(4))  # -1: second should rank higher (0 > -5) → correct
        assert result.total.item() < 1e-3  # with norm: |−5| = 5 > 0 → wrong order → large loss


class TestRankNetCriterion:
    def test_confident_correct_low_loss(self) -> None:
        """score_first >> score_second and target=1 → P≈1 → loss ≈ 0."""
        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        B, D = 4, 8
        logits = torch.stack([torch.ones(B, D) * 5.0, torch.zeros(B, D)], dim=1)  # [B, 2, D]
        result = crit(logits, torch.ones(B))  # 1: first preferred
        assert result.total.item() < 1e-3

    def test_confident_wrong_high_loss(self) -> None:
        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        B, D = 4, 8
        logits = torch.stack([torch.ones(B, D) * 5.0, torch.zeros(B, D)], dim=1)
        result = crit(logits, torch.zeros(B))  # 0: second preferred, but first scores far higher
        assert result.total.item() > 1.0

    def test_tie_target_is_log_two(self) -> None:
        """Equal scores + tie target 0.5 → BCE(gap=0, 0.5) = log 2."""
        from math import log

        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        logits = torch.zeros(4, 2, 8)  # equal scores → gap 0
        result = crit(logits, torch.full((4,), 0.5))
        assert abs(result.total.item() - log(2)) < 1e-5

    def test_loss_components_key(self) -> None:
        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        result = crit(torch.randn(2, 2, 8), torch.ones(2))
        assert "ranknet" in result.components

    def test_wrong_n_views_raises(self) -> None:
        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        with pytest.raises(ValueError, match="2"):
            crit(torch.randn(4, 3, 8), torch.ones(4))  # n_views=3, not 2

    def test_scalar_head_uses_raw_signed_score(self) -> None:
        """D=1: the raw signed scalar is the score (norm would drop the sign)."""
        from src.losses.ranking import RankNetCriterion

        crit = RankNetCriterion()
        logits = torch.zeros(4, 2, 1)
        logits[:, 0, 0] = -10.0  # first score −10 → P(first≻second) = sigmoid(−10) ≈ 0
        result = crit(logits, torch.zeros(4))  # target 0: second preferred → correct → loss ≈ 0
        assert result.total.item() < 1e-3  # with norm: sigmoid(|−10|−0) = sigmoid(10) ≈ 1, BCE(·, 0) huge


# ---------------------------------------------------------------------------
# Objective strategy
# ---------------------------------------------------------------------------


class TestMetricObjective:
    def test_metric_objective_registered(self) -> None:
        from src.tasks import Objective, objective_strategies

        assert Objective.METRIC in objective_strategies

    def test_metric_objective_supports_ranking(self) -> None:
        from src.tasks import Topology
        from src.tasks.strategies.objective import MetricObjective

        obj = MetricObjective()
        assert obj.supports(Topology.RANKING)
        assert not obj.supports(Topology.GLOBAL)
        assert not obj.supports(Topology.DENSE)

    def test_metric_objective_out_features_passthrough(self) -> None:
        from src.tasks.strategies.objective import MetricObjective

        assert MetricObjective().out_features(64) == 64


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------


class TestRankingTaskBuilder:
    def test_metric_x_ranking_valid(self) -> None:
        from src.tasks import TaskBuilder
        from src.tasks.strategies.objective import MetricObjective
        from src.tasks.strategies.topology import RankingTopology

        task = TaskBuilder(
            topology=RankingTopology(view_keys=("anchor", "positive", "negative")),
            objective=MetricObjective(),
        ).build("sim", num_classes=32)
        assert task.name == "sim"
        assert task.head_spec.view_keys == ("anchor", "positive", "negative")

    def test_non_ranking_objective_on_ranking_topology_raises(self) -> None:
        from src.tasks import MulticlassObjective, TaskBuilder
        from src.tasks.strategies.topology import RankingTopology

        builder = TaskBuilder(
            topology=RankingTopology(view_keys=("a", "b")),
            objective=MulticlassObjective(),
        )
        with pytest.raises(ValueError, match="ranking"):
            builder.build("sim", num_classes=3)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestRankingPresets:
    def test_triplet_in_task_presets_registry(self) -> None:
        from src.tasks import task_presets

        assert "triplet" in task_presets

    def test_triplet_preset_builds_task_without_view_keys(self) -> None:
        """Presets do not bake in view_keys — wiring derives them from data.inputs."""
        from src.tasks.presets import triplet

        task = triplet("visual_sim", num_classes=64)
        assert task.name == "visual_sim"
        assert task.head_spec.view_keys is None

    def test_pairwise_ranking_in_task_presets_registry(self) -> None:
        from src.tasks import task_presets

        assert "pairwise_ranking" in task_presets

    def test_pairwise_ranking_preset_builds_task_without_view_keys(self) -> None:
        from src.tasks.presets import pairwise_ranking

        task = pairwise_ranking("pair_sim", num_classes=32)
        assert task.name == "pair_sim"
        assert task.head_spec.view_keys is None

    def test_triplet_preset_default_loss_is_triplet_margin(self) -> None:
        """The preset (not the objective) carries the default loss."""
        from src.losses.ranking import TripletMarginCriterion
        from src.tasks.presets import triplet

        assert isinstance(triplet("sim", num_classes=64).criterion, TripletMarginCriterion)

    def test_pairwise_ranking_preset_default_loss_is_margin_ranking(self) -> None:
        """A 2-view pair defaults to margin_ranking, not the 3-view triplet loss."""
        from src.losses.ranking import MarginRankingCriterion
        from src.tasks.presets import pairwise_ranking

        assert isinstance(pairwise_ranking("pair", num_classes=32).criterion, MarginRankingCriterion)

    def test_input_aliases_multi_view(self) -> None:
        """input_aliases extracts dict keys in declaration order."""
        from src.core.keys import IMAGE
        from src.data.loaders import input_aliases

        assert input_aliases({"anchor": "a", "positive": "p", "negative": "n"}) == (
            "anchor",
            "positive",
            "negative",
        )
        assert input_aliases("image_path") == (IMAGE,)

    def test_ranking_task_head_spec_gets_view_keys_after_wiring(self) -> None:
        """After build_tasks the RANKING head_spec carries view_keys from data.inputs."""
        import dataclasses

        from src.data.loaders import input_aliases
        from src.tasks.presets import triplet

        # Simulate what build_tasks does for a single RANKING task.
        task = triplet("sim", num_classes=64)
        assert task.head_spec.view_keys is None  # pre-wiring

        view_keys = input_aliases({"anchor": "a_path", "positive": "p_path", "negative": "n_path"})
        task = dataclasses.replace(task, head_spec=dataclasses.replace(task.head_spec, view_keys=view_keys))
        assert task.head_spec.view_keys == ("anchor", "positive", "negative")
