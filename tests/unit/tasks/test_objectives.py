"""``TaskObjective`` contracts: out_features, bricks, and the num_classes flag."""

from __future__ import annotations

import pytest
import torch

from src.core import Objective, TargetFacts
from src.losses import InfoNceCriterion
from src.tasks import (
    BinaryObjective,
    ContinuousObjective,
    MetricObjective,
    MulticlassObjective,
    MultilabelObjective,
)
from src.tasks.registry import objective_registry


def test_registry_covers_the_implemented_objectives() -> None:
    assert set(objective_registry) == {
        Objective.MULTICLASS,
        Objective.BINARY,
        Objective.MULTILABEL,
        Objective.CONTINUOUS,
        Objective.METRIC,
    }


def test_out_features_follow_label_semantics() -> None:
    assert MulticlassObjective().out_features(TargetFacts(num_classes=10)) == 10
    assert MultilabelObjective().out_features(TargetFacts(num_classes=5)) == 5
    assert BinaryObjective().out_features(TargetFacts()) == 1
    assert ContinuousObjective().out_features(TargetFacts()) == 1


def test_objectives_declare_whether_they_need_num_classes() -> None:
    assert MulticlassObjective.needs_num_classes is True
    assert MultilabelObjective.needs_num_classes is True
    assert BinaryObjective.needs_num_classes is False
    assert ContinuousObjective.needs_num_classes is False
    assert MetricObjective.needs_num_classes is False


def test_only_the_metric_objective_has_no_target_adapter() -> None:
    """No target column means there is nothing to adapt: the brick is absent."""
    assert MetricObjective().build_target_adapter(TargetFacts()) is None
    assert MulticlassObjective().build_target_adapter(TargetFacts()) is not None
    assert ContinuousObjective().build_target_adapter(TargetFacts()) is not None


def test_metric_kwargs_carry_the_label_semantics() -> None:
    assert MulticlassObjective().metric_kwargs(TargetFacts(num_classes=3)) == {"task": "multiclass", "num_classes": 3}
    assert BinaryObjective().metric_kwargs(TargetFacts()) == {"task": "binary"}
    assert MultilabelObjective().metric_kwargs(TargetFacts(num_classes=5)) == {"task": "multilabel", "num_labels": 5}
    assert ContinuousObjective().metric_kwargs(TargetFacts()) == {}


def test_metric_bricks_default_to_infonce_over_raw_embeddings() -> None:
    objective = MetricObjective()
    carrier = torch.randn(4, 2, 8)

    criterion = objective.build_criterion(TargetFacts())
    predictions = objective.build_activation(TargetFacts())(carrier)

    assert isinstance(criterion, InfoNceCriterion)
    assert torch.equal(predictions, carrier)


def test_multiclass_out_features_requires_num_classes() -> None:
    with pytest.raises(LookupError, match="num_classes"):
        MulticlassObjective().out_features(TargetFacts())


def test_multiclass_bricks_work_together() -> None:
    objective = MulticlassObjective()
    logits = torch.randn(4, 3)

    adapted = objective.build_target_adapter(TargetFacts())(torch.tensor([0, 1, 2, 0]))
    loss = objective.build_criterion(TargetFacts())(logits, adapted.for_loss)
    probabilities = objective.build_activation(TargetFacts())(logits)

    assert set(loss.parts) == {"ce"}
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_binary_adapter_floats_the_loss_view_only() -> None:
    adapted = BinaryObjective().build_target_adapter(TargetFacts())(torch.tensor([0, 1, 1]))

    assert adapted.for_loss.dtype == torch.float32
    assert adapted.for_metrics.dtype == torch.long


def test_binary_activation_squeezes_the_single_logit() -> None:
    probabilities = BinaryObjective().build_activation(TargetFacts())(torch.zeros(4, 1))

    assert probabilities.shape == (4,)
    assert torch.allclose(probabilities, torch.full((4,), 0.5))


def test_continuous_activation_squeezes_single_output_heads() -> None:
    predictions = ContinuousObjective().build_activation(TargetFacts())(torch.tensor([[1.0], [2.0]]))

    assert torch.equal(predictions, torch.tensor([1.0, 2.0]))


def test_binary_activation_squeezes_the_channel_on_dense_logits() -> None:
    probabilities = BinaryObjective().build_activation(TargetFacts())(torch.zeros(2, 1, 8, 8))

    assert probabilities.shape == (2, 8, 8)


def test_multiclass_bricks_work_on_dense_logits() -> None:
    objective = MulticlassObjective()
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))

    loss = objective.build_criterion(TargetFacts())(logits, target)
    probabilities = objective.build_activation(TargetFacts())(logits)

    assert loss.total.shape == ()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, 8, 8))
