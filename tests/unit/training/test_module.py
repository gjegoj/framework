"""The Lightning module: a step becomes a number to descend, an epoch becomes a report.

The learner here answers with what the batch already carries, so nothing below is a test about a
network — what is under test is the bookkeeping the loop needs: which stage owns which metric state,
what reaches the log and what reaches a tracker, and what the optimizer is built over.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, cast

import lightning as L
import pytest
import torch
from lightning.pytorch.loggers import Logger
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from src.core import Batch, LossOutput, Matrix, StepOutput, TargetInfo, require_tensor
from src.metrics.build import build_metrics
from src.tasks import Classification, Task
from src.training import Learner, TrainingModule
from src.training.base import FitProfile, ParameterGroup
from tests.support.models import Echo

CLASSES = {0: "cat", 1: "dog"}
TARGET = torch.tensor([0, 1])
RIGHT = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
WRONG = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
TERM = "entropy"
METRICS = {"accuracy": {"name": "accuracy"}, "confusion_matrix": {"name": "confusion_matrix"}}


def batch(logits: Tensor) -> Batch:
    """A batch whose input *is* the prediction: the learner below hands it straight back."""
    return Batch(inputs={"species": logits}, targets={"species": TARGET}, count=2)


class Ready(Dataset[Batch]):
    """Batches that need no collation, so a loop can be driven by the values a test wrote."""

    def __init__(self, batches: Sequence[Batch]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> Batch:
        return self.batches[index]


def loader(*logits: Tensor) -> DataLoader[Batch]:
    return DataLoader(Ready([batch(one) for one in logits]), batch_size=1, collate_fn=lambda ready: ready[0])


class Answers(Learner):
    """A learner that answers with what the batch carries, weighing each task as the run declared."""

    def __init__(
        self, tasks: Mapping[str, Task], *, answered: Sequence[str] | None = None, carries_loss: bool = True
    ) -> None:
        super().__init__(Echo({name: RIGHT for name in tasks}), tasks)
        self.scale = nn.Parameter(torch.ones(()))
        self.answered = tuple(tasks) if answered is None else tuple(answered)
        self.carries_loss = carries_loss

    def step(self, batch: Batch) -> StepOutput:
        predictions = {name: require_tensor(batch.inputs[name], name=name) * self.scale for name in self.answered}
        targets = {name: require_tensor(batch.targets[name], name=name) for name in self.answered}
        return StepOutput(
            loss=self._loss(predictions) if self.carries_loss else None, predictions=predictions, targets=targets
        )

    def _loss(self, predictions: Mapping[str, Tensor]) -> LossOutput:
        """One term per task, weighed and namespaced exactly as the standard learner does."""
        terms = []
        for name, value in predictions.items():
            reported = value.mean()
            term = LossOutput(reported, losses={TERM: reported}, contributions={TERM: reported})
            terms.append((term * self.tasks[name].weight).prefixed(name))
        return reduce(operator.add, terms) if terms else LossOutput(self.scale * 0.0)


class Recorder(Logger):
    """A Lightning logger that can also draw: the tracker port, on a real trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.scalars: dict[str, float] = {}
        self.drawn: list[tuple[str, Matrix, int]] = []

    @property
    def name(self) -> str:
        return "recorder"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.scalars.update(metrics)

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        self.drawn.append((title, matrix, iteration))


def plain(groups: Sequence[ParameterGroup]) -> list[dict[str, Any]]:
    """torch takes group dicts, and a TypedDict is one — a distinction only a type checker draws."""
    return [cast("dict[str, Any]", group) for group in groups]


def task(weight: float = 1.0) -> dict[str, Task]:
    return {"species": Classification("species", TargetInfo(classes=CLASSES), weight=weight)}


def module(
    tasks: Mapping[str, Task] | None = None,
    *,
    measured: bool = True,
    scheduler: Any = None,
    optimizer: Any = None,
    **learner_options: Any,
) -> TrainingModule:
    declared = task() if tasks is None else tasks
    metrics = {name: build_metrics(METRICS, one.facts()) for name, one in declared.items() if measured}
    return TrainingModule(
        Answers(declared, **learner_options),
        optimizer_factory=optimizer or (lambda groups: SGD(plain(groups), lr=0.1)),
        scheduler_factory=scheduler,
        metrics=metrics,
    )


def trainer(**options: Any) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **{"logger": False, "max_epochs": 1, "num_sanity_val_steps": 0, **options},
    )


class TestReporting:
    def test_the_objective_is_reported_as_a_total_and_the_terms_it_is_made_of(self) -> None:
        fit = trainer()
        fit.fit(module(), train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        assert {"train/loss", f"train/species/{TERM}", "val/loss", f"val/species/{TERM}"} <= set(fit.logged_metrics)

    def test_a_weighted_task_reports_its_share_of_the_objective_beside_the_term_itself(self) -> None:
        """A term stays comparable between runs; the share is what the total is made of."""
        fit, weighted = trainer(), trainer()
        fit.fit(module(), train_dataloaders=loader(RIGHT))
        weighted.fit(module(task(weight=0.5)), train_dataloaders=loader(RIGHT))

        share = f"train/species/{TERM}/contribution"
        assert share not in fit.logged_metrics
        assert weighted.logged_metrics[share] == pytest.approx(weighted.logged_metrics[f"train/species/{TERM}"] / 2)

    def test_each_stage_scores_the_data_it_was_given_and_no_other(self) -> None:
        fit = trainer()
        fit.fit(module(), train_dataloaders=loader(RIGHT), val_dataloaders=loader(WRONG))

        assert fit.logged_metrics["train/species/accuracy"] == 1.0
        assert fit.logged_metrics["val/species/accuracy"] == 0.0

    def test_a_stage_starts_every_epoch_from_nothing(self) -> None:
        """A second evaluation reads its own rows: state kept across epochs would average the two."""
        under_test = module()
        trainer().fit(under_test, train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        again = trainer()
        again.validate(under_test, dataloaders=loader(WRONG))

        assert again.logged_metrics["val/species/accuracy"] == 0.0

    def test_a_reading_that_means_a_picture_reaches_the_tracker_and_the_numbers_reach_the_log(self) -> None:
        recorder = Recorder()
        fit = trainer(logger=recorder)

        fit.fit(module(), train_dataloaders=loader(RIGHT))

        titles = [title for title, _, _ in recorder.drawn]
        assert titles == ["train/species/confusion_matrix"]
        assert isinstance(recorder.drawn[0][1], Matrix)
        assert "train/species/accuracy" in recorder.scalars
        assert "train/species/confusion_matrix" not in recorder.scalars

    def test_a_sanity_check_draws_nothing_and_reports_nothing(self) -> None:
        """Lightning suppresses `self.log` during one; a picture handed straight to a tracker needs the same."""
        recorder = Recorder()
        fit = trainer(logger=recorder, num_sanity_val_steps=1)

        fit.fit(module(), train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        drawn = sorted(title for title, _, _ in recorder.drawn)
        assert drawn == ["train/species/confusion_matrix", "val/species/confusion_matrix"], (
            "one picture per stage that ran an epoch, and none for the batches the check looked at"
        )


class TestOptimization:
    def test_the_optimizer_is_built_over_the_groups_the_learner_owns(self) -> None:
        seen: list[ParameterGroup] = []

        def factory(groups: Sequence[ParameterGroup]) -> Optimizer:
            seen.extend(groups)
            return SGD(plain(groups), lr=0.1)

        trainer().fit(module(optimizer=factory), train_dataloaders=loader(RIGHT))

        assert [group["name"] for group in seen] == ["backbone", "species"]

    def test_a_schedule_is_sized_by_the_fit_the_trainer_turned_out_to_be(self) -> None:
        sized: list[FitProfile] = []

        def factory(optimizer: Optimizer, profile: FitProfile) -> dict[str, Any]:
            sized.append(profile)
            return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)}

        trainer(max_epochs=2).fit(module(scheduler=factory), train_dataloaders=loader(RIGHT, WRONG))

        assert sized == [FitProfile(total_steps=4, epochs=2)]

    def test_a_fit_with_no_declared_end_cannot_size_a_schedule(self) -> None:
        def factory(optimizer: Optimizer, profile: FitProfile) -> dict[str, Any]:
            raise AssertionError("a schedule must not be built over a fit of unknown length")

        with pytest.raises(ValueError, match="epochs"):
            trainer(max_epochs=-1, max_steps=2).fit(module(scheduler=factory), train_dataloaders=loader(RIGHT))


class TestAssembly:
    def test_metrics_for_a_task_the_learner_never_heard_of_are_refused(self) -> None:
        with pytest.raises(ValueError, match="weather"):
            TrainingModule(
                Answers(task()),
                optimizer_factory=lambda groups: SGD(plain(groups), lr=0.1),
                metrics={"weather": build_metrics(METRICS, task()["species"].facts())},
            )

    def test_a_task_the_learner_did_not_answer_for_is_named_rather_than_scored_on_nothing(self) -> None:
        silent = module(task(), answered=())

        with pytest.raises(ValueError, match="species"):
            silent.training_step(batch(RIGHT), 0)

    def test_a_training_step_that_produced_no_loss_has_nothing_to_descend(self) -> None:
        with pytest.raises(ValueError, match="loss"):
            module(carries_loss=False).training_step(batch(RIGHT), 0)

    def test_evaluation_may_report_metrics_without_a_loss_at_all(self) -> None:
        """Declared by `StepOutput.loss` being optional: a run may be scored without being learned."""
        run = trainer()
        run.validate(module(carries_loss=False), dataloaders=loader(RIGHT))

        assert "val/loss" not in run.logged_metrics
        assert run.logged_metrics["val/species/accuracy"] == 1.0
