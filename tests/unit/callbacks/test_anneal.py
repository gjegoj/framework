"""``AnnealCriterion``: a criterion's number follows the schedule, wherever it lives."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import pytest
import torch
from torch import nn

from src.callbacks import AnnealCriterion
from src.callbacks.anneal import SCHEDULES, scheduled_value
from src.callbacks.registry import callback_registry
from src.core import Criterion, Loss
from src.losses import CrossEntropyCriterion, WeightedSumCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.activations import softmax_probabilities
from src.tasks.adapters import as_class_indices
from tests.support.fakes import FlattenBackbone
from tests.support.lightning import quiet_trainer


class AnnealedRun(L.LightningModule):
    """The little a schedule needs: ``model.criteria`` keyed by task, and steps."""

    def __init__(self, criterion: Criterion) -> None:
        super().__init__()
        self.model = CompositeModel(
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

    def training_step(self, batch: Any, index: int) -> torch.Tensor:
        logits = self.model.heads["label"](batch[0].flatten(1))
        total: torch.Tensor = self.model.criteria["label"](logits, batch[1]).total
        return total

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def fit(callback: AnnealCriterion, criterion: Criterion, epochs: int = 4) -> None:
    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(2, 4, 1, 1), torch.tensor([0, 1])), batch_size=2
    )
    quiet_trainer(max_epochs=epochs, callbacks=[callback]).fit(AnnealedRun(criterion), data)


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
    with pytest.raises(ValueError, match="label"):
        fit(AnnealCriterion(task="mask", parameter="gamma", start=0.0, end=1.0), CrossEntropyCriterion())


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
