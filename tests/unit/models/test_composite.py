"""``CompositeModel``: the backbone × heads family behind the unified ``Model`` port."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.core import AdaptedTarget, Batch, Criterion, Loss, Model
from src.models import CompositeModel, LinearHead, TaskComponents
from tests.support.fakes import FlattenBackbone
from tests.support.narrowing import tensor


class ConstantCriterion(Criterion):
    """Returns a fixed loss value — makes weighting arithmetic exactly checkable."""

    def __init__(self, name: str, value: float) -> None:
        super().__init__()
        self._name = name
        self._value = value

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.part(self._name, torch.tensor(self._value))


class ScaledL1Criterion(Criterion):
    """Carries a learnable parameter — proves criteria land in the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.part("l1", (self.scale * (logits.squeeze(-1) - target)).abs().mean())


def identity_adapter(target: Tensor) -> AdaptedTarget:
    return AdaptedTarget(for_loss=target, for_metrics=target)


def make_batch() -> Batch:
    torch.manual_seed(0)
    return Batch(
        inputs={"image": torch.randn(4, 3, 2, 2)},
        targets={"label": torch.tensor([0, 1, 2, 0]), "score": torch.rand(4)},
    )


def make_model() -> CompositeModel:
    return CompositeModel(
        backbone=FlattenBackbone(dim=12),
        components={
            "label": TaskComponents(
                head=LinearHead(12, 3),
                criterion=ConstantCriterion("ce", 1.0),
                activation=lambda logits: logits.argmax(dim=1),
                target_adapter=identity_adapter,
            ),
            "score": TaskComponents(
                head=LinearHead(12, 1),
                criterion=ScaledL1Criterion(),
                activation=lambda logits: logits.squeeze(-1),
                target_adapter=identity_adapter,
                weight=0.5,
            ),
        },
    )


def test_composite_model_implements_the_model_port() -> None:
    assert isinstance(make_model(), Model)


def test_step_returns_scoped_loss_and_predictions_from_one_forward() -> None:
    loss, prediction, targets = make_model().step(make_batch())

    assert set(loss.parts) == {"label/ce", "score/l1"}
    assert loss.total.requires_grad
    assert set(prediction.outputs) == {"label", "score"}
    assert tensor(prediction.outputs["label"]).shape == (4,)
    assert prediction.features is not None
    assert set(targets) == {"label", "score"}


def test_step_targets_carry_the_metric_view() -> None:
    batch = make_batch()

    result = make_model().step(batch)

    assert torch.equal(tensor(result.targets["label"]), tensor(batch.targets["label"]))


def test_step_applies_task_weights_to_the_total() -> None:
    model = CompositeModel(
        backbone=FlattenBackbone(dim=12),
        components={
            "a": TaskComponents(
                head=LinearHead(12, 3),
                criterion=ConstantCriterion("const", 1.0),
                activation=lambda logits: logits,
                target_adapter=identity_adapter,
            ),
            "b": TaskComponents(
                head=LinearHead(12, 3),
                criterion=ConstantCriterion("const", 1.0),
                activation=lambda logits: logits,
                target_adapter=identity_adapter,
                weight=0.5,
            ),
        },
    )
    batch = Batch(
        inputs={"image": torch.randn(2, 3, 2, 2)},
        targets={"a": torch.tensor([0, 1]), "b": torch.tensor([0, 1])},
    )

    loss, _, _ = model.step(batch)

    assert loss.total.item() == pytest.approx(1.5)
    assert loss.parts["b/const"].item() == pytest.approx(0.5)


def test_predict_does_not_need_targets() -> None:
    batch = Batch(inputs={"image": torch.randn(2, 3, 2, 2)}, targets={})

    prediction = make_model().predict(batch)

    assert set(prediction.outputs) == {"label", "score"}


def test_step_names_the_task_when_a_target_is_missing() -> None:
    batch = Batch(inputs={"image": torch.randn(2, 3, 2, 2)}, targets={})

    with pytest.raises(LookupError, match="label"):
        make_model().step(batch)


def test_heads_and_criteria_are_registered_for_checkpointing() -> None:
    state = make_model().state_dict()

    assert "heads.label._projection.weight" in state
    assert "criteria.score.scale" in state


def test_composite_model_requires_components() -> None:
    with pytest.raises(ValueError, match="component"):
        CompositeModel(backbone=FlattenBackbone(dim=12), components={})


def test_a_component_without_a_target_adapter_is_structure_supervised() -> None:
    """No target adapter means no target lookup: the step runs on empty targets."""
    model = CompositeModel(
        backbone=FlattenBackbone(dim=12),
        components={
            "pair": TaskComponents(
                head=LinearHead(12, 8),
                criterion=ConstantCriterion("pairwise", 1.0),
                activation=lambda logits: logits,
                target_adapter=None,
            )
        },
    )
    batch = Batch(inputs={"image": torch.randn(2, 3, 2, 2)}, targets={})

    loss, _, targets = model.step(batch)

    assert loss.total.shape == ()
    assert tensor(targets["pair"]).numel() == 0


class RecordingCriterion(Criterion):
    """Keeps the target it was handed, so a test can see what really arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.received: Tensor | None = None

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        self.received = target
        return Loss.part("recorded", logits.sum() * 0)


def test_a_target_with_no_adapter_is_delivered_raw() -> None:
    """No adapter means nothing to shape — a ranking pair's preference must still arrive."""
    recorder = RecordingCriterion()
    model = CompositeModel(
        backbone=FlattenBackbone(dim=12),
        components={
            "prefers": TaskComponents(
                head=LinearHead(12, 1),
                criterion=recorder,
                activation=lambda logits: logits,
                target_adapter=None,
            )
        },
    )
    preference = torch.tensor([1.0, -1.0, 0.5, 1.0])

    model.step(Batch(inputs={"image": torch.randn(4, 3, 2, 2)}, targets={"prefers": preference}))

    assert recorder.received is not None and torch.equal(recorder.received, preference)


def test_a_task_without_a_target_stays_structure_supervised() -> None:
    """`absent` is only for a target that truly is — InfoNCE-style tasks lose nothing."""
    recorder = RecordingCriterion()
    model = CompositeModel(
        backbone=FlattenBackbone(dim=12),
        components={
            "views": TaskComponents(
                head=LinearHead(12, 2),
                criterion=recorder,
                activation=lambda logits: logits,
                target_adapter=None,
            )
        },
    )

    model.step(Batch(inputs={"image": torch.randn(4, 3, 2, 2)}, targets={}))

    assert recorder.received is not None and recorder.received.numel() == 0


def test_the_model_hands_back_the_values_before_its_activation() -> None:
    """Distillation and calibration compare pre-activation values, and an activation cannot be undone.

    The 'label' task activates with argmax, which discards the scores outright:
    what a temperature would scale is reachable only if the model offers it.
    """
    prediction = make_model().predict(make_batch())

    assert prediction.logits is not None
    assert prediction.logits["label"].shape == (4, 3)
    assert tensor(prediction.outputs["label"]).shape == (4,)
    assert torch.equal(prediction.logits["label"].argmax(dim=1), tensor(prediction.outputs["label"]))


def test_a_step_reports_the_same_logits_it_scored() -> None:
    """The soft loss of a distilled run is computed from the student's own logits, so a step has to expose them."""
    result = make_model().step(make_batch())

    assert result.prediction.logits is not None
    assert set(result.prediction.logits) == set(result.prediction.outputs)
    assert result.prediction.logits["label"].shape == (4, 3)
