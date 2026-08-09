"""``AnnealCriterion``: a criterion's number follows the schedule, wherever it lives."""

from __future__ import annotations

import logging
from typing import Any, cast

import lightning as L
import pytest
import torch
from torch import nn

from src.callbacks import AnnealCriterion
from src.callbacks.anneal import SCHEDULES, scheduled_value
from src.callbacks.registry import callback_registry
from src.core import Batch, Criterion, Loss, Model, Prediction, StepResult
from src.losses import CrossEntropyCriterion, KLDivergenceCriterion, WeightedSumCriterion
from src.models import CompositeModel, DistilledModel, LinearHead, TaskComponents, without_teachers
from src.tasks.activations import softmax_probabilities
from src.tasks.adapters import as_class_indices
from tests.support.fakes import FlattenBackbone
from tests.support.lightning import quiet_trainer


def composed(criterion: Criterion) -> CompositeModel:
    """One classification task on one backbone — what ``build_model`` assembles."""
    return CompositeModel(
        backbone=FlattenBackbone(dim=4),
        components={
            "label": TaskComponents(
                head=LinearHead(4, 3),
                criterion=criterion,
                activation=softmax_probabilities,
                target_adapter=as_class_indices,
            )
        },
    )


class _WholeModel(Model):
    """A family that arrives whole: its loss is internal, so it composes no task's brick.

    The port's default ``criterion_of`` answers ``None`` for exactly this shape.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.trunk(images.flatten(1)))

    def step(self, batch: Batch) -> StepResult:
        raise NotImplementedError("This test drives the module's own training_step.")

    def predict(self, batch: Batch) -> Prediction:
        raise NotImplementedError("This test drives the module's own training_step.")


class AnnealedRun(L.LightningModule):
    """The little a schedule needs: a model carrying the task's criterion, and steps.

    ``distilled`` wraps the student exactly as the ``distillation:`` section does, so
    the callback meets the module tree a real distilled run hands it rather than a
    shape only this test has.
    """

    def __init__(self, criterion: Criterion, distilled: bool = False) -> None:
        super().__init__()
        student = composed(criterion)
        self.model: Model = (
            DistilledModel(
                student=student,
                teachers=[composed(CrossEntropyCriterion())],
                criterion=KLDivergenceCriterion(),
            )
            if distilled
            else student
        )

    def training_step(self, batch: Any, index: int) -> torch.Tensor:
        student = cast("CompositeModel", without_teachers(self.model))
        logits = student.heads["label"](batch[0].flatten(1))
        total: torch.Tensor = student.criteria["label"](logits, batch[1]).total
        return total

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def fit(callback: AnnealCriterion, criterion: Criterion, epochs: int = 4, distilled: bool = False) -> None:
    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2, 4, 1, 1), torch.tensor([0, 1])), batch_size=2
    )
    quiet_trainer(max_epochs=epochs, callbacks=[callback]).fit(AnnealedRun(criterion, distilled), data)


def test_the_number_moves_from_start_to_end_over_the_run() -> None:
    criterion = CrossEntropyCriterion(label_smoothing=0.0)

    fit(AnnealCriterion(task="label", parameter="label_smoothing", start=0.2, end=0.0), criterion)

    assert criterion._loss.label_smoothing == pytest.approx(0.0)


def test_start_overrides_the_constructed_value_at_epoch_zero() -> None:
    assert scheduled_value(0, 4, 0.2, 0.0, SCHEDULES["linear"]) == pytest.approx(0.2)


def test_the_end_holds_once_the_window_is_over() -> None:
    """``over: 0.5`` finishes the ramp halfway through and stays there."""
    assert scheduled_value(9, 5, 0.0, 2.0, SCHEDULES["cosine"]) == pytest.approx(2.0)


def test_a_part_prefix_picks_one_criterion_of_a_composite() -> None:
    ce = CrossEntropyCriterion(label_smoothing=0.0)
    composite = WeightedSumCriterion([(ce, 1.0)])

    fit(AnnealCriterion(task="label", parameter="ce.label_smoothing", start=0.3, end=0.3), composite)

    assert ce._loss.label_smoothing == pytest.approx(0.3)


def test_a_distilled_run_still_reaches_the_criterion_of_its_task() -> None:
    """``distillation:`` nests the student, and a schedule has to follow it there.

    Both sections are supported and documented, and neither says the other is excluded.
    But the callback read ``model.criteria``, which only the composite family has, so a
    run declaring both died at ``on_fit_start`` — before a single batch, with a message
    about a model that "exposes none" rather than about the two features not composing.

    ``backbone_path`` in assembly already accounts for exactly this nesting, so the
    knowledge existed in the codebase; this reader simply did not have it.
    """
    criterion = CrossEntropyCriterion(label_smoothing=0.9)

    fit(
        AnnealCriterion(task="label", parameter="label_smoothing", start=0.2, end=0.0),
        criterion,
        distilled=True,
    )

    assert criterion._loss.label_smoothing == pytest.approx(0.0)


def test_an_ambiguous_name_lists_the_parts_that_carry_it() -> None:
    class OtherSmoothed(CrossEntropyCriterion):
        part_name = "other"

    composite = WeightedSumCriterion(
        [(CrossEntropyCriterion(label_smoothing=0.1), 1.0), (OtherSmoothed(label_smoothing=0.2), 1.0)]
    )

    with pytest.raises(ValueError, match="ce.label_smoothing"):
        fit(AnnealCriterion(task="label", parameter="label_smoothing", start=0.0, end=0.0), composite)


def test_a_learnable_parameter_is_refused() -> None:
    """Writing over an nn.Parameter every epoch silently fights the optimizer."""

    class Learnable(Criterion):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Loss:
            return Loss.part("learnable", self.scale * logits.sum())

    with pytest.raises(ValueError, match="optimizer"):
        fit(AnnealCriterion(task="label", parameter="scale", start=0.0, end=1.0), Learnable())


def test_a_missing_attribute_lists_the_numeric_ones() -> None:
    """The message must let a user discover what is schedulable without reading torch sources."""
    with pytest.raises(ValueError, match="ignore_index"):
        fit(AnnealCriterion(task="label", parameter="gamma", start=0.0, end=1.0), CrossEntropyCriterion())


def test_an_unknown_task_lists_the_configured_ones() -> None:
    """The refusal now comes from the model, which is the thing that knows its tasks.

    ``LookupError`` rather than ``ValueError``: an unknown key naming the known ones is
    what ``Registry.get``, ``Features[...]`` and ``DataModule.dataset`` already raise.
    """
    with pytest.raises(LookupError, match="label"):
        fit(AnnealCriterion(task="mask", parameter="gamma", start=0.0, end=1.0), CrossEntropyCriterion())


def test_a_family_that_composes_no_criterion_is_told_so_by_name() -> None:
    """A vendor family owns its loss internally, so there is no per-task brick to move.

    The port's default answer is ``None``, and the schedule turns that into a sentence
    naming the family — rather than the ``AttributeError`` a tree walk would have raised.
    """

    class VendorRun(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = _WholeModel()

        def training_step(self, batch: Any, index: int) -> torch.Tensor:
            # nn.Module.__call__ erases the return type to Any; pin it back.
            return cast("torch.Tensor", self.model(batch[0])).sum()

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2, 4, 1, 1), torch.tensor([0, 1])), batch_size=2
    )
    callback = AnnealCriterion(task="label", parameter="gamma", start=0.0, end=1.0)

    with pytest.raises(ValueError, match="composes none"):
        quiet_trainer(callbacks=[callback]).fit(VendorRun(), data)


@pytest.mark.parametrize("over", [0.0, 1.5])
def test_a_window_outside_the_run_is_refused(over: float) -> None:
    with pytest.raises(ValueError, match="over"):
        AnnealCriterion(task="label", parameter="gamma", start=0.0, end=1.0, over=over)


def test_an_unknown_easing_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="cosine"):
        AnnealCriterion(task="label", parameter="gamma", start=0.0, end=1.0, schedule="quadratic")


def test_it_is_reachable_from_config_by_name() -> None:
    built = callback_registry.create("anneal", task="label", parameter="gamma", start=0.0, end=2.0)

    assert isinstance(built, AnnealCriterion)


def test_the_schedule_announces_itself_the_way_every_other_boundary_does(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ramp is a pure function of the epoch and leaves no trace of its own.

    A run whose loss ends up configured differently from how it started would
    otherwise say nothing about why — and this is the one callback with a schedule
    that printed nothing at all. The moment is given in both currencies, as every
    other boundary in `callbacks/` gives it.
    """
    criterion = CrossEntropyCriterion(label_smoothing=0.9)
    callback = AnnealCriterion(task="label", parameter="label_smoothing", start=0.2, end=0.0, over=0.5)

    with caplog.at_level(logging.INFO):
        fit(callback, criterion, epochs=4)

    (said,) = [record.getMessage() for record in caplog.records if "Annealing" in record.getMessage()]
    assert "label_smoothing of task 'label' from 0.2 to 0.0" in said
    assert "reaching it at epoch 1 (step 1)" in said  # window of 2 epochs, one step each
