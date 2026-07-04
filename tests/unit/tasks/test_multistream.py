"""Tests for M7b-A: multi-encoder contrastive machinery (MULTISTREAM topology).

TDD order: RED → GREEN → REFACTOR.
All classes/functions imported here must exist after the GREEN phase.

The "duality" of CLIP/SIGLIP lives only in the loss; the backbone, extractor and
topology are N-general.  These tests exercise the N=2 path (the only loss we ship
in sub-project A) plus the N-general backbone/extractor on a 3-encoder case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from src.models.assembly import CompositeModel

if TYPE_CHECKING:
    from src.models.backbones.multi import MultiEncoderBackbone


# ---------------------------------------------------------------------------
# Multi-encoder backbone
# ---------------------------------------------------------------------------


class TestMultiEncoderBackbone:
    def _backbone(self, embed_dim: int | None = 32) -> "MultiEncoderBackbone":
        from src.models.backbones import EmbeddingBackbone
        from src.models.backbones.multi import MultiEncoderBackbone

        return MultiEncoderBackbone(
            encoders={
                "left": EmbeddingBackbone(embedding_dim=8),
                "right": EmbeddingBackbone(embedding_dim=16),
            },
            embed_dim=embed_dim,
        )

    def test_each_encoder_produces_a_namespaced_stream(self) -> None:
        backbone = self._backbone(embed_dim=32)
        inputs = {"left": torch.randn(4, 8), "right": torch.randn(4, 16)}
        bundle = backbone(inputs)
        assert bundle["left"].shape == (4, 32)
        assert bundle["right"].shape == (4, 32)

    def test_feature_dim_returns_embed_dim(self) -> None:
        backbone = self._backbone(embed_dim=32)
        assert backbone.feature_dim("left") == 32
        assert backbone.feature_dim("right") == 32

    def test_passthrough_when_embed_dim_none(self) -> None:
        """embed_dim=None → no projection; streams keep their encoder dims."""
        backbone = self._backbone(embed_dim=None)
        inputs = {"left": torch.randn(4, 8), "right": torch.randn(4, 16)}
        bundle = backbone(inputs)
        assert bundle["left"].shape == (4, 8)
        assert bundle["right"].shape == (4, 16)
        assert backbone.feature_dim("left") == 8
        assert backbone.feature_dim("right") == 16

    def test_encoders_have_separate_weights(self) -> None:
        """Same input through two same-arch encoders → different output (separate projections)."""
        from src.models.backbones import EmbeddingBackbone
        from src.models.backbones.multi import MultiEncoderBackbone

        backbone = MultiEncoderBackbone(
            encoders={"a": EmbeddingBackbone(embedding_dim=8), "b": EmbeddingBackbone(embedding_dim=8)},
            embed_dim=32,
        )
        x = torch.randn(4, 8)
        bundle = backbone({"a": x, "b": x})
        assert not torch.allclose(bundle["a"], bundle["b"])

    def test_supports_n_greater_than_two(self) -> None:
        """Backbone is N-general — three encoders work identically."""
        from src.models.backbones import EmbeddingBackbone
        from src.models.backbones.multi import MultiEncoderBackbone

        backbone = MultiEncoderBackbone(
            encoders={name: EmbeddingBackbone(embedding_dim=8) for name in ("img", "txt", "audio")},
            embed_dim=16,
        )
        inputs = {name: torch.randn(4, 8) for name in ("img", "txt", "audio")}
        bundle = backbone(inputs)
        assert set(bundle.streams) == {"img", "txt", "audio"}
        assert all(bundle[name].shape == (4, 16) for name in inputs)

    def test_unknown_stream_raises(self) -> None:
        backbone = self._backbone(embed_dim=32)
        with pytest.raises(KeyError):
            backbone.feature_dim("missing")


# ---------------------------------------------------------------------------
# Multi-stream forward (extractor + identity head, via CompositeModel)
# ---------------------------------------------------------------------------


class TestMultiStreamForward:
    def _model(self, stream_keys: tuple[str, ...], emb_dim: int = 32) -> "CompositeModel":
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.models.backbones.multi import MultiEncoderBackbone
        from src.tasks.strategies.topology import MultistreamTopology

        backbone = MultiEncoderBackbone(
            encoders={key: EmbeddingBackbone(embedding_dim=8) for key in stream_keys},
            embed_dim=emb_dim,
        )
        spec = MultistreamTopology(stream_keys=stream_keys).head_spec(out_features=emb_dim)
        return build_composite_model(backbone, {"align": spec})

    def test_two_stream_output_shape(self) -> None:
        B, D = 4, 32
        model = self._model(("left", "right"), emb_dim=D)
        inputs = {"left": torch.randn(B, 8), "right": torch.randn(B, 8)}
        out = model(inputs)
        assert out.task_logits["align"].shape == (B, 2, D)

    def test_three_stream_output_shape(self) -> None:
        B, D = 4, 16
        model = self._model(("img", "txt", "audio"), emb_dim=D)
        inputs = {name: torch.randn(B, 8) for name in ("img", "txt", "audio")}
        out = model(inputs)
        assert out.task_logits["align"].shape == (B, 3, D)

    def test_stream_order_preserved(self) -> None:
        """Stacking follows stream_keys order, not dict iteration luck."""
        B, D = 2, 16
        model = self._model(("left", "right"), emb_dim=D)
        inputs = {"left": torch.zeros(B, 8), "right": torch.ones(B, 8)}
        out = model(inputs)
        logits = out.task_logits["align"]
        # left (zeros) → projection bias only; right (ones) → bias + weights·1.
        # They must differ, confirming the two streams are not collapsed.
        assert not torch.allclose(logits[:, 0], logits[:, 1])


# ---------------------------------------------------------------------------
# Identity head
# ---------------------------------------------------------------------------


class TestIdentityHead:
    def test_identity_head_returns_input_unchanged(self) -> None:
        from src.models.registry import head_builders

        head = head_builders.create("identity", in_features=10, out_features=10)
        x = torch.randn(4, 2, 10)
        assert torch.equal(head(x), x)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


class TestMultistreamTopology:
    def test_registered(self) -> None:
        from src.tasks import Topology, topology_strategies

        assert Topology.MULTISTREAM in topology_strategies

    def test_head_spec_with_explicit_stream_keys(self) -> None:
        from src.tasks.strategies.topology import MultistreamTopology

        spec = MultistreamTopology(stream_keys=("left", "right")).head_spec(out_features=64)
        assert spec.kind == "identity"
        assert spec.stream_keys == ("left", "right")
        assert spec.out_features == 64

    def test_default_has_no_stream_keys(self) -> None:
        """stream_keys=None by default — wiring derives them from data.inputs."""
        from src.tasks.strategies.topology import MultistreamTopology

        topo = MultistreamTopology()
        assert topo.stream_keys is None
        assert topo.head_spec(out_features=32).stream_keys is None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class TestMetricObjective:
    def test_registered(self) -> None:
        from src.tasks import Objective, objective_strategies

        assert Objective.METRIC in objective_strategies

    def test_supports_multistream_and_ranking(self) -> None:
        """One metric objective covers both embedding topologies plus GLOBAL proxy classification."""
        from src.tasks import Topology
        from src.tasks.strategies.objective import MetricObjective

        obj = MetricObjective()
        assert obj.supports(Topology.MULTISTREAM)
        assert obj.supports(Topology.MULTIVIEW)
        assert obj.supports(Topology.GLOBAL)  # arcface_embedding: one image -> one embedding
        assert not obj.supports(Topology.DENSE)

    def test_out_features_passthrough(self) -> None:
        from src.tasks.strategies.objective import MetricObjective

        assert MetricObjective().out_features(64) == 64


# ---------------------------------------------------------------------------
# InfoNCE criterion
# ---------------------------------------------------------------------------


class TestInfoNCECriterion:
    def test_aligned_pairs_have_low_loss(self) -> None:
        """Matching rows across streams (orthogonal between samples) → near-zero loss."""
        from src.losses.contrastive import InfoNCECriterion

        crit = InfoNCECriterion()
        z = torch.eye(4, 8)  # rows are orthonormal basis vectors
        logits = torch.stack([z, z], dim=1)  # [4, 2, 8], stream0 == stream1
        result = crit(logits, torch.zeros(4))
        assert result.total.item() < 0.05

    def test_misaligned_pairs_have_higher_loss(self) -> None:
        from src.losses.contrastive import InfoNCECriterion

        crit = InfoNCECriterion()
        z0 = torch.eye(4, 8)
        z1 = torch.roll(z0, shifts=1, dims=0)  # positive sits off the diagonal
        aligned = crit(torch.stack([z0, z0], dim=1), torch.zeros(4)).total
        misaligned = crit(torch.stack([z0, z1], dim=1), torch.zeros(4)).total
        assert misaligned.item() > aligned.item()
        assert misaligned.item() > 1.0

    def test_loss_components_key(self) -> None:
        from src.losses.contrastive import InfoNCECriterion

        crit = InfoNCECriterion()
        logits = torch.randn(3, 2, 8)
        result = crit(logits, torch.zeros(3))
        assert "info_nce" in result.components

    def test_temperature_is_learnable(self) -> None:
        from src.losses.contrastive import InfoNCECriterion

        crit = InfoNCECriterion()
        learnable = [p for p in crit.parameters() if p.requires_grad]
        assert len(learnable) == 1  # the logit_scale

    def test_wrong_n_views_raises(self) -> None:
        from src.losses.contrastive import InfoNCECriterion

        crit = InfoNCECriterion()
        with pytest.raises(ValueError, match="2"):
            crit(torch.randn(4, 3, 8), torch.zeros(4))  # N=3, not 2


# ---------------------------------------------------------------------------
# SigLIP criterion (sigmoid pairwise loss — sibling of InfoNCE)
# ---------------------------------------------------------------------------


class TestSigLIPCriterion:
    def test_registered_under_siglip(self) -> None:
        import src.losses.contrastive  # noqa: F401 — import registers the criterion
        from src.losses.registry import criteria

        assert "siglip" in criteria

    def test_strong_alignment_has_low_loss(self) -> None:
        """logit_scale=20, bias=-10 → diagonal logit +10, off-diagonal -10 → near-zero loss."""
        from src.losses.contrastive import SigLIPCriterion

        crit = SigLIPCriterion(logit_scale=20.0, bias=-10.0)
        z = torch.eye(4, 8)  # orthonormal rows
        logits = torch.stack([z, z], dim=1)  # [4, 2, 8], stream0 == stream1
        result = crit(logits, torch.zeros(4))
        assert result.total.item() < 0.05

    def test_misaligned_pairs_have_higher_loss(self) -> None:
        from src.losses.contrastive import SigLIPCriterion

        crit = SigLIPCriterion()
        z0 = torch.eye(4, 8)
        z1 = torch.roll(z0, shifts=1, dims=0)  # positive sits off the diagonal
        aligned = crit(torch.stack([z0, z0], dim=1), torch.zeros(4)).total
        misaligned = crit(torch.stack([z0, z1], dim=1), torch.zeros(4)).total
        assert misaligned.item() > aligned.item()

    def test_loss_components_key(self) -> None:
        from src.losses.contrastive import SigLIPCriterion

        crit = SigLIPCriterion()
        result = crit(torch.randn(3, 2, 8), torch.zeros(3))
        assert "siglip" in result.components

    def test_scale_and_bias_are_learnable(self) -> None:
        """SigLIP has TWO learnable parameters (scale + bias), unlike InfoNCE."""
        from src.losses.contrastive import SigLIPCriterion

        crit = SigLIPCriterion()
        learnable = [p for p in crit.parameters() if p.requires_grad]
        assert len(learnable) == 2

    def test_wrong_n_views_raises(self) -> None:
        from src.losses.contrastive import SigLIPCriterion

        crit = SigLIPCriterion()
        with pytest.raises(ValueError, match="2"):
            crit(torch.randn(4, 3, 8), torch.zeros(4))  # N=3, not 2


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------


class TestContrastivePreset:
    def test_in_task_presets_registry(self) -> None:
        from src.tasks import task_presets

        assert "contrastive" in task_presets

    def test_builds_task_without_stream_keys(self) -> None:
        """Preset bakes in no stream_keys — wiring derives them from data.inputs."""
        from src.tasks.presets import contrastive

        task = contrastive("align", num_classes=64)
        assert task.name == "align"
        assert task.head_spec.stream_keys is None
        assert task.head_spec.kind == "identity"

    def test_preset_default_loss_is_info_nce(self) -> None:
        """The default loss lives on the preset now, not the (shared) metric objective."""
        from src.losses.contrastive import InfoNCECriterion
        from src.tasks.presets import contrastive

        assert isinstance(contrastive("align", num_classes=64).criterion, InfoNCECriterion)

    def test_non_contrastive_objective_on_multistream_raises(self) -> None:
        from src.tasks import MulticlassObjective, TaskBuilder
        from src.tasks.strategies.topology import MultistreamTopology

        builder = TaskBuilder(
            topology=MultistreamTopology(stream_keys=("a", "b")),
            objective=MulticlassObjective(),
        )
        with pytest.raises(ValueError, match="multistream"):
            builder.build("align", num_classes=3)


# ---------------------------------------------------------------------------
# Wiring: stream_keys derived from data.inputs (parallel to view_keys)
# ---------------------------------------------------------------------------


class TestContrastiveWiring:
    def test_stream_keys_derived_from_data_inputs(self) -> None:
        import dataclasses

        from src.data.loaders import input_aliases
        from src.tasks.presets import contrastive

        task = contrastive("align", num_classes=64)
        assert task.head_spec.stream_keys is None  # pre-wiring

        stream_keys = input_aliases({"left": "left_path", "right": "right_path"})
        task = dataclasses.replace(task, head_spec=dataclasses.replace(task.head_spec, stream_keys=stream_keys))
        assert task.head_spec.stream_keys == ("left", "right")
