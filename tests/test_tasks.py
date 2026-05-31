"""Unit tests for the task-composition layer and its loss/metric bricks."""

import pytest
import torch

from src.core.entities import HeadSpec, Task
from src.core.enums import Stage
from src.losses.criterion import CrossEntropyCriterion
from src.metrics.builders import build_classification_metrics
from src.tasks import (
    GlobalTopology,
    MulticlassObjective,
    MulticlassTaskCodec,
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
        assert spec == HeadSpec(kind="linear", out_features=7, feature_key="pooled")

    def test_multiclass_objective_bricks(self) -> None:
        objective = MulticlassObjective()
        assert objective.out_features(5) == 5
        assert objective.supports(Topology.GLOBAL)
        assert not objective.supports(Topology.EMBEDDING)
        assert isinstance(objective.build_task_codec(), MulticlassTaskCodec)

    def test_strategies_registered(self) -> None:
        assert Topology.GLOBAL in topology_strategies
        assert Objective.MULTICLASS in objective_strategies


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

    def test_metric_set_update_compute_reset(self) -> None:
        metrics = build_classification_metrics(3, task="multiclass")
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

    def test_invalid_combination_raises(self) -> None:
        class _EmbeddingTopology(TopologyStrategy):
            kind = Topology.EMBEDDING

            def head_spec(self, out_features: int) -> HeadSpec:
                return HeadSpec(kind="linear", out_features=out_features)

        with pytest.raises(ValueError, match="not supported on topology 'embedding'"):
            TaskBuilder(_EmbeddingTopology(), MulticlassObjective()).build("x", num_classes=2)


class TestClassificationPreset:
    def test_default_is_multiclass(self) -> None:
        task = classification("label", num_classes=5)
        assert task.head_spec.out_features == 5

    def test_preset_resolvable_via_registry(self) -> None:
        task = task_presets.create("classification", name="x", num_classes=2)
        assert task.name == "x"

    def test_unknown_objective_raises(self) -> None:
        with pytest.raises(ValueError):
            classification("label", num_classes=3, objective="bogus")
