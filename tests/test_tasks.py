"""Unit tests for the task-composition layer and its loss/metric bricks."""

import pytest
import torch

from src.core.entities import HeadSpec, Task
from src.core.enums import Stage
from src.losses.criterion import (
    BCEWithLogitsCriterion,
    CrossEntropyCriterion,
    DiceCriterion,
    L1Criterion,
    MSECriterion,
    WeightedSumCriterion,
)
from src.metrics.builders import build_metric_set
from src.tasks import (
    BinaryObjective,
    BinaryTaskCodec,
    ContinuousObjective,
    ContinuousTaskCodec,
    GlobalTopology,
    MulticlassObjective,
    MulticlassTaskCodec,
    MultilabelObjective,
    MultilabelTaskCodec,
    Objective,
    TaskBuilder,
    Topology,
    classification,
    objective_strategies,
    task_presets,
    topology_strategies,
)
from src.tasks.strategies.topology import TopologyStrategy


class TestTaxonomyAndStrategies:
    def test_global_topology_head_spec(self) -> None:
        spec = GlobalTopology().head_spec(out_features=7)
        assert spec.kind == "linear" and spec.out_features == 7 and spec.feature_key == "pooled"
        assert spec.prefer_native is True

    def test_multiclass_objective_bricks(self) -> None:
        objective = MulticlassObjective()
        assert objective.out_features(5) == 5
        assert objective.supports(Topology.GLOBAL)
        assert not objective.supports(Topology.RANKING)
        assert isinstance(objective.build_task_codec(), MulticlassTaskCodec)

    def test_binary_objective_bricks(self) -> None:
        obj = BinaryObjective()
        assert obj.out_features(2) == 1  # always 1 regardless of num_classes
        assert obj.supports(Topology.GLOBAL)
        assert isinstance(obj.build_task_codec(), BinaryTaskCodec)

    def test_multilabel_objective_bricks(self) -> None:
        obj = MultilabelObjective()
        assert obj.out_features(7) == 7
        assert obj.supports(Topology.GLOBAL)
        assert isinstance(obj.build_task_codec(), MultilabelTaskCodec)

    def test_continuous_objective_bricks(self) -> None:
        obj = ContinuousObjective()
        assert obj.out_features(1) == 1
        assert obj.supports(Topology.GLOBAL)
        assert isinstance(obj.build_task_codec(), ContinuousTaskCodec)

    def test_dense_topology_head_spec(self) -> None:
        from src.tasks import DenseTopology

        spec = DenseTopology().head_spec(out_features=4)
        assert spec.kind == "conv" and spec.out_features == 4 and spec.feature_key == "decoder"
        assert spec.prefer_native is True

    def test_strategies_registered(self) -> None:
        assert Topology.GLOBAL in topology_strategies
        assert Topology.DENSE in topology_strategies
        assert Objective.MULTICLASS in objective_strategies
        assert Objective.BINARY in objective_strategies
        assert Objective.MULTILABEL in objective_strategies
        assert Objective.CONTINUOUS in objective_strategies


class TestBricks:
    def test_task_codec_squeezes_and_longs(self) -> None:
        view = MulticlassTaskCodec().adapt(torch.tensor([[0], [2]], dtype=torch.int32))
        assert view.loss.shape == (2,)
        assert view.loss.dtype == torch.long
        assert torch.equal(view.loss, view.metric)

    def test_cross_entropy_backprops(self) -> None:
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.randint(0, 3, (4,))
        result = CrossEntropyCriterion()(logits, target)
        assert result.total.ndim == 0
        assert "cross_entropy" in result.components
        result.total.backward()
        assert logits.grad is not None

    def test_bce_binary_backprops(self) -> None:
        logits = torch.randn(4, 1, requires_grad=True)
        target = torch.randint(0, 2, (4, 1)).float()
        result = BCEWithLogitsCriterion()(logits, target)
        assert result.total.ndim == 0
        assert "bce" in result.components
        result.total.backward()
        assert logits.grad is not None

    def test_bce_multilabel_backprops(self) -> None:
        logits = torch.randn(4, 5, requires_grad=True)
        target = torch.randint(0, 2, (4, 5)).float()
        result = BCEWithLogitsCriterion()(logits, target)
        assert result.total.ndim == 0
        result.total.backward()

    def test_mse_backprops(self) -> None:
        logits = torch.randn(4, 1, requires_grad=True)
        target = torch.randn(4, 1)
        result = MSECriterion()(logits, target)
        assert result.total.ndim == 0
        assert "mse" in result.components
        result.total.backward()
        assert logits.grad is not None

    def test_l1_backprops(self) -> None:
        logits = torch.randn(4, 1, requires_grad=True)
        target = torch.randn(4, 1)
        result = L1Criterion()(logits, target)
        assert result.total.ndim == 0
        assert "l1" in result.components
        result.total.backward()

    def test_dice_multiclass_backprops(self) -> None:
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)  # [B,C,H,W]
        target = torch.randint(0, 4, (2, 8, 8))  # [B,H,W]
        result = DiceCriterion(mode="multiclass")(logits, target)
        assert result.total.ndim == 0
        assert "dice" in result.components
        result.total.backward()
        assert logits.grad is not None

    def test_weighted_sum_ce_dice_combines_and_backprops(self) -> None:
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (2, 8, 8))
        crit = WeightedSumCriterion(losses={"cross_entropy": 1.0, "dice": 0.5})
        result = crit(logits, target)
        assert {"cross_entropy", "dice"} <= set(result.components)
        result.total.backward()
        assert logits.grad is not None

    def test_weighted_sum_empty_losses_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one loss"):
            WeightedSumCriterion(losses={})

    def test_metric_set_update_compute_reset(self) -> None:
        metrics = build_metric_set(None, base_kwargs={"task": "multiclass", "num_classes": 3})
        preds = torch.randn(8, 3).softmax(dim=1)
        target = torch.randint(0, 3, (8,))
        metrics.update(preds, target)
        computed = metrics.compute()
        assert "accuracy" in computed
        metrics.reset()


class TestTaskBuilder:
    def test_build_produces_full_task_with_per_stage_metrics(self) -> None:
        task = TaskBuilder(GlobalTopology(), MulticlassObjective()).build("label", num_classes=3)
        assert isinstance(task, Task)
        assert task.head_spec.out_features == 3
        assert task.feature_key == "pooled"
        assert set(task.metrics) == {Stage.TRAIN, Stage.VAL, Stage.TEST}
        # Each stage gets its own metric instance (independent state).
        assert task.metrics[Stage.TRAIN] is not task.metrics[Stage.VAL]

    # Bridge proof: same GlobalTopology, different objectives → different assemblies.
    def test_bridge_binary_on_global_topology(self) -> None:
        task = TaskBuilder(GlobalTopology(), BinaryObjective()).build("is_cat", num_classes=2)
        assert task.head_spec.out_features == 1
        assert task.head_spec.feature_key == "pooled"
        assert isinstance(task.codec, BinaryTaskCodec)

    def test_bridge_multilabel_on_global_topology(self) -> None:
        task = TaskBuilder(GlobalTopology(), MultilabelObjective()).build("tags", num_classes=5)
        assert task.head_spec.out_features == 5
        assert isinstance(task.codec, MultilabelTaskCodec)
        # default metrics should be accuracy (multilabel accuracy)
        preds = torch.sigmoid(torch.randn(8, 5))
        targets = torch.randint(0, 2, (8, 5))
        task.metrics[Stage.TRAIN].update(preds, targets)
        assert "accuracy" in task.metrics[Stage.TRAIN].compute()

    def test_bridge_continuous_on_global_topology(self) -> None:
        task = TaskBuilder(GlobalTopology(), ContinuousObjective()).build("value", num_classes=1)
        assert task.head_spec.out_features == 1
        assert isinstance(task.codec, ContinuousTaskCodec)
        # default metrics should be mse + mae
        preds = torch.randn(8, 1)
        targets = torch.randn(8, 1)
        task.metrics[Stage.TRAIN].update(preds, targets)
        computed = task.metrics[Stage.TRAIN].compute()
        assert {"mse", "mae"} <= set(computed)

    def test_bridge_dense_multiclass_reuses_objective(self) -> None:
        from src.tasks import DenseTopology

        # Same MulticlassObjective as classification, new DenseTopology → segmentation task.
        task = TaskBuilder(DenseTopology(), MulticlassObjective()).build("mask", num_classes=5)
        assert task.head_spec.kind == "conv"
        assert task.head_spec.feature_key == "decoder"
        assert task.head_spec.out_features == 5
        assert isinstance(task.codec, MulticlassTaskCodec)  # objective bricks unchanged

    def test_invalid_combination_raises(self) -> None:
        class _RankingTopology(TopologyStrategy):
            kind = Topology.RANKING

            def head_spec(self, out_features: int) -> HeadSpec:
                return HeadSpec(kind="linear", out_features=out_features)

        with pytest.raises(ValueError, match="not supported on topology 'ranking'"):
            TaskBuilder(_RankingTopology(), MulticlassObjective()).build("x", num_classes=2)


class TestRegressionPreset:
    def test_regression_builds_continuous_task(self) -> None:
        from src.tasks import regression

        task = regression("price", num_classes=1)
        assert task.head_spec.out_features == 1
        assert isinstance(task.codec, ContinuousTaskCodec)

    def test_regression_default_metric_is_mae(self) -> None:
        from src.tasks import regression

        task = regression("price", num_classes=1)
        preds = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        task.metrics[Stage.TRAIN].update(preds, targets)
        assert {"mae"} <= set(task.metrics[Stage.TRAIN].compute())

    def test_preset_carries_topology_and_default_objective(self) -> None:
        from src.tasks.taxonomy import Objective, Topology

        preset = task_presets.create("regression")
        assert preset.topology == Topology.GLOBAL
        assert preset.default_objective == Objective.CONTINUOUS

    def test_preset_resolves_objective_override(self) -> None:
        from src.tasks.taxonomy import Objective

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
        assert isinstance(task.codec, MulticlassTaskCodec)

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
        assert preset.default_codec == "mask"
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


class TestConfigDrivenBricks:
    def test_loss_override_applies_params(self) -> None:
        task = classification(
            "label",
            num_classes=3,
            loss={"name": "cross_entropy", "label_smoothing": 0.1},
        )
        assert isinstance(task.criterion, CrossEntropyCriterion)
        assert task.criterion._loss.label_smoothing == pytest.approx(0.1)

    def test_loss_string_spec_selects_default(self) -> None:
        task = classification("label", num_classes=3, loss="cross_entropy")
        assert isinstance(task.criterion, CrossEntropyCriterion)

    def test_metrics_spec_builds_named_collection_with_params(self) -> None:
        task = classification(
            "label",
            num_classes=4,
            metrics={"accuracy": None, "macro_f1": {"name": "f1", "average": "macro"}},
        )
        preds = torch.randn(8, 4).softmax(dim=1)
        target = torch.randint(0, 4, (8,))
        task.metrics[Stage.TRAIN].update(preds, target)
        computed = task.metrics[Stage.TRAIN].compute()
        assert {"accuracy", "macro_f1"} <= set(computed)

    def test_classification_default_metrics(self) -> None:
        task = classification("label", num_classes=3)
        preds = torch.randn(8, 3).softmax(dim=1)
        task.metrics[Stage.TRAIN].update(preds, torch.randint(0, 3, (8,)))
        keys = set(task.metrics[Stage.TRAIN].compute())
        assert {"precision", "recall", "f1", "confusion_matrix"} <= keys


class TestFeatureKeyOverride:
    def test_builder_feature_key_override_changes_head_spec(self) -> None:
        task = TaskBuilder(GlobalTopology(), MulticlassObjective()).build(
            "label", num_classes=3, feature_key_override="encoder_last"
        )
        assert task.head_spec.feature_key == "encoder_last"

    def test_preset_feature_key_overrides_topology_default(self) -> None:
        task = classification("label", num_classes=5, feature_key="encoder_last")
        assert task.head_spec.feature_key == "encoder_last"
        # prefer_native stays on: native_head("encoder_last") → ClassificationHead
        assert task.head_spec.prefer_native is True

    def test_feature_key_none_keeps_topology_default(self) -> None:
        task = classification("label", num_classes=3, feature_key=None)
        assert task.head_spec.feature_key == "pooled"

    def test_segmentation_feature_key_override(self) -> None:
        from src.tasks import segmentation

        task = segmentation("mask", num_classes=4, feature_key="decoder")
        assert task.head_spec.feature_key == "decoder"  # same as default, just explicit

    def test_feature_key_override_does_not_disable_native_head(self) -> None:
        # feature_key_override changes the key but leaves prefer_native untouched.
        task = TaskBuilder(GlobalTopology(), MulticlassObjective()).build(
            "label", num_classes=3, feature_key_override="encoder_last"
        )
        assert task.head_spec.prefer_native is True

    def test_feature_key_override_before_head_override(self) -> None:
        # Both supplied: feature_key is in place when head_override runs.
        task = TaskBuilder(GlobalTopology(), MulticlassObjective()).build(
            "label",
            num_classes=3,
            feature_key_override="encoder_last",
            head_override="linear",
        )
        assert task.head_spec.feature_key == "encoder_last"
        assert task.head_spec.prefer_native is False


class TestTaskCarriesAxes:
    def test_classification_task_has_global_multiclass_axes(self) -> None:
        from src.tasks.presets import classification
        from src.tasks.taxonomy import Objective, Topology

        task = classification("label", num_classes=3)
        assert task.topology == Topology.GLOBAL
        assert task.objective == Objective.MULTICLASS
