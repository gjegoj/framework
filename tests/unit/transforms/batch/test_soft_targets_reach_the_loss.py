"""What a batch transform writes must survive the model step that reads it.

The transform tests either side of this seam in isolation: they check the shape
a mix produces, and the wiring test checks the callback is built. Neither runs
the batch through a task, which is where the two halves have to agree.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from src.assembly.metrics import build_metric_sets
from src.config import MetricConfig
from src.core import AdaptedTarget, Batch, DataProfile, Objective, OutputTopology, Stage, TargetFacts, Task
from src.losses import CrossEntropyCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.registry import objective_registry
from src.transforms.batch import CutMix, MixUp, Mosaic
from tests.support.fakes import FlattenBackbone
from tests.support.narrowing import tensor

CLASSES = 3
SIDE = 2
FEATURES = 3 * SIDE * SIDE


def task(objective: Objective = Objective.MULTICLASS) -> Task:
    return Task(name="label", output_topology=OutputTopology.GLOBAL, objective=objective, metrics={})


def profile() -> DataProfile:
    built = DataProfile()
    built.record("label", TargetFacts(num_classes=CLASSES))
    return built


def batch() -> Batch:
    torch.manual_seed(0)
    return Batch(
        inputs={"image": torch.randn(4, 3, SIDE, SIDE)},
        targets={"label": torch.tensor([0, 1, 2, 0])},
    )


def adapt(objective: Objective, target: torch.Tensor, facts: TargetFacts | None = None) -> AdaptedTarget:
    """Apply the adapter the objective itself chooses, so no test can pick a kinder one."""
    adapter = objective_registry.create(objective).build_target_adapter(facts or TargetFacts(num_classes=CLASSES))
    assert adapter is not None
    return adapter(target)


def model(objective: Objective = Objective.MULTICLASS) -> CompositeModel:
    """The bricks the objective itself chooses, so the test cannot pick a kinder adapter."""
    behaviour = objective_registry.create(objective)
    facts = TargetFacts(num_classes=CLASSES)
    return CompositeModel(
        backbone=FlattenBackbone(dim=FEATURES),
        components={
            "label": TaskComponents(
                # Every objective this file parametrises over projects onto classes; a
                # metric one would answer None and belongs to an identity head instead.
                head=LinearHead(FEATURES, cast("int", behaviour.out_features(facts))),
                criterion=CrossEntropyCriterion(),
                activation=behaviour.build_activation(facts),
                target_adapter=behaviour.build_target_adapter(facts),
            )
        },
    )


@pytest.mark.parametrize("transform", ["mixup", "cutmix", "mosaic"])
def test_a_mixed_batch_trains(transform: str) -> None:
    """The whole point of the transform: a gradient that reflects both samples."""
    mixing = {"mixup": MixUp, "cutmix": CutMix, "mosaic": Mosaic}[transform]
    torch.manual_seed(0)
    built = model()

    result = built.step(mixing([task()], profile())(batch()))
    result.loss.total.backward()

    assert result.loss.total.item() > 0.0
    gradients = [p.grad for p in built.parameters() if p.grad is not None]
    assert gradients and any(bool(g.abs().sum() > 0) for g in gradients)


def test_the_distribution_reaches_the_loss_unrounded() -> None:
    """Rounding it would make every mixed target the same class, and mostly class zero."""
    mixed = MixUp([task()], profile())(batch())

    adapted = adapt(Objective.MULTICLASS, tensor(mixed.targets["label"]))

    assert adapted.for_loss.is_floating_point()
    assert torch.allclose(adapted.for_loss.sum(dim=1), torch.ones(4))


def test_metrics_still_get_one_class_per_sample() -> None:
    """torchmetrics ranks against a class, not a distribution over them."""
    mixed = MixUp([task()], profile())(batch())

    adapted = adapt(Objective.MULTICLASS, tensor(mixed.targets["label"]))

    assert adapted.for_metrics.shape == (4,)
    assert not adapted.for_metrics.is_floating_point()


def test_an_ordinary_batch_is_untouched_by_the_fix() -> None:
    """Class indices are still class indices; the two forms are told apart, not merged."""
    plain = batch()

    adapted = adapt(Objective.MULTICLASS, tensor(plain.targets["label"]))

    assert torch.equal(adapted.for_loss, tensor(plain.targets["label"]))
    assert torch.equal(adapted.for_metrics, tensor(plain.targets["label"]))


@pytest.mark.parametrize(
    "mixed",
    [torch.tensor([0.7, 0.2, 0.9, 0.1]), torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])],
    ids=["binary", "multilabel"],
)
def test_a_mixed_indicator_reaches_its_metric_as_a_label(mixed: torch.Tensor) -> None:
    """Binary cross-entropy takes the share, but torchmetrics rejects anything but 0/1."""
    objective = Objective.BINARY if mixed.dim() == 1 else Objective.MULTILABEL

    adapted = adapt(objective, mixed)

    assert adapted.for_loss.is_floating_point()
    assert set(adapted.for_metrics.unique().tolist()) <= {0, 1}


def test_a_metric_accepts_what_the_adapter_produced() -> None:
    """Asserted against the real metric the objective picks, not a stand-in."""
    sets = build_metric_sets(
        Objective.MULTILABEL, facts=TargetFacts(num_classes=2), metrics={"f1": MetricConfig(name="f1")}
    )
    mixed = torch.tensor([[0.7, 0.3], [0.2, 0.8]])

    adapted = adapt(Objective.MULTILABEL, mixed, TargetFacts(num_classes=2))
    sets[Stage.TRAIN].update(torch.rand(2, 2), adapted.for_metrics)

    assert sets[Stage.TRAIN].compute()


def test_a_continuous_target_is_never_rounded() -> None:
    """A price has no hard form; rounding one to 0/1 would destroy the target."""
    prices = torch.tensor([1.0, 3.7, 62.4])

    adapted = adapt(Objective.CONTINUOUS, prices, TargetFacts())

    assert torch.equal(adapted.for_metrics, prices)
