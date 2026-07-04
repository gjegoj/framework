"""Task presets (classification/regression/segmentation/arcface family) and activations."""

from __future__ import annotations

import pytest
import torch

from src.core.enums import Stage
from src.tasks import (
    ContinuousTargetAdapter,
    MulticlassTargetAdapter,
    Topology,
    classification,
    task_presets,
)


class TestRegressionPreset:
    def test_regression_builds_continuous_task(self) -> None:
        from src.tasks import regression

        task = regression("price", num_classes=1)
        assert task.head_spec.out_features == 1
        assert isinstance(task.adapter, ContinuousTargetAdapter)

    def test_regression_default_metric_is_mae(self) -> None:
        from src.tasks import regression

        task = regression("price", num_classes=1)
        preds = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        task.metrics[Stage.TRAIN].update(preds, targets)
        assert {"mae"} <= set(task.metrics[Stage.TRAIN].compute())

    def test_preset_carries_topology_and_default_objective(self) -> None:
        from src.core.taxonomy import Objective, Topology

        preset = task_presets.create("regression")
        assert preset.topology == Topology.GLOBAL
        assert preset.default_objective == Objective.CONTINUOUS

    def test_preset_resolves_objective_override(self) -> None:
        from src.core.taxonomy import Objective

        preset = task_presets.create("classification")
        assert preset.resolve_objective(None) == Objective.MULTICLASS
        assert preset.resolve_objective("binary") == Objective.BINARY


class TestSegmentationPreset:
    def test_builds_dense_multiclass_task(self) -> None:
        from src.tasks import segmentation

        task = segmentation("mask", num_classes=4)
        assert task.head_spec.kind == "conv"
        assert task.head_spec.feature_key == "decoder"
        assert task.head_spec.out_features == 4
        assert isinstance(task.adapter, MulticlassTargetAdapter)

    def test_segmentation_default_metrics(self) -> None:
        from src.tasks import segmentation

        task = segmentation("mask", num_classes=3)
        preds = torch.randn(2, 3, 8, 8).softmax(dim=1)  # [B,C,H,W]
        target = torch.randint(0, 3, (2, 8, 8))  # [B,H,W]
        task.metrics[Stage.TRAIN].update(preds, target)
        keys = set(task.metrics[Stage.TRAIN].compute())
        assert {"iou", "f1", "precision", "recall"} <= keys

    def test_preset_carries_mask_codec_default(self) -> None:
        preset = task_presets.create("segmentation")
        assert preset.default_encoder == "mask"
        assert preset.topology == Topology.DENSE


class TestClassificationPreset:
    def test_default_is_multiclass(self) -> None:
        task = classification("label", num_classes=5)
        assert task.head_spec.out_features == 5

    def test_preset_resolvable_via_registry(self) -> None:
        task = task_presets.create("classification").build(name="x", num_classes=2)
        assert task.name == "x"

    def test_unknown_objective_raises(self) -> None:
        with pytest.raises(ValueError):
            classification("label", num_classes=3, objective="bogus")


class TestNormalizeActivation:
    def test_output_is_unit_norm(self) -> None:
        from src.tasks.activations import NormalizeActivation

        embedding = torch.randn(4, 16) * 7.0
        normalized = NormalizeActivation()(embedding)
        assert normalized.shape == embedding.shape
        assert torch.allclose(normalized.norm(dim=-1), torch.ones(4), atol=1e-6)


class TestArcFacePresets:
    def test_arcface_classifier_defaults(self) -> None:
        from src.losses.angular import ArcFaceCriterion
        from src.tasks.presets import task_presets

        task = task_presets.create("arcface")("species", num_classes=3)
        assert task.head_spec.kind == "cosine"
        assert task.head_spec.prefer_native is False
        assert isinstance(task.criterion, ArcFaceCriterion)

    def test_arcface_user_head_override_wins(self) -> None:
        from src.tasks.presets import task_presets

        task = task_presets.create("arcface")("species", num_classes=3, head="linear")
        assert task.head_spec.kind == "linear"

    def test_arcface_embedding_defaults(self) -> None:
        from src.losses.angular import ProxyAngularCriterion
        from src.tasks.activations import NormalizeActivation
        from src.tasks.presets import task_presets

        task = task_presets.create("arcface_embedding")("embed", num_classes=32, class_count=5)
        assert task.head_spec.kind == "linear"
        assert task.head_spec.prefer_native is False
        assert task.head_spec.out_features == 32
        assert isinstance(task.activation, NormalizeActivation)
        assert isinstance(task.criterion, ProxyAngularCriterion)
        assert task.criterion.prototypes.shape == (32, 5)

    def test_arcface_embedding_encoder_default_is_label(self) -> None:
        from src.tasks.presets import task_presets

        assert task_presets.create("arcface_embedding").default_encoder == "label"

    def test_arcface_embedding_without_class_count_raises(self) -> None:
        from src.tasks.presets import task_presets

        with pytest.raises(ValueError, match="target"):
            task_presets.create("arcface_embedding")("embed", num_classes=32)
