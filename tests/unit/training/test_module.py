"""The Lightning module: a step becomes a number to descend, an epoch becomes a report.

The learner here answers with what the batch already carries, so nothing below is a test about a
network — what is under test is the bookkeeping the loop needs: which stage owns which metric state,
what reaches the log and what reaches a tracker, and what the optimizer is built over.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, ClassVar, cast

import lightning as L
import pytest
import torch
import torchmetrics
from lightning.pytorch.loggers import Logger
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset

from src.core import Batch, LossOutput, Matrix, Stage, StepOutput, TargetInfo, require_tensor
from src.metrics.build import build_metrics
from src.tasks import Classification, Task
from src.training import Learner, TrainingModule
from src.training.base import FitProfile, ParameterGroup, StepPreview
from src.training.module import module_at
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


class WhenStill(torchmetrics.Metric):
    """A reading that means nothing while the model is still moving, as ranking against a gallery does.

    Counts its own updates, so a stage it should never be read on is visible as a number rather than
    only as an absence.
    """

    read_on: ClassVar[frozenset[Stage]] = frozenset({Stage.VAL, Stage.TEST})
    higher_is_better = True
    seen: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("seen", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.seen += 1

    def compute(self) -> Tensor:
        return self.seen


def plain(groups: Sequence[ParameterGroup]) -> list[dict[str, Any]]:
    """torch takes group dicts, and a TypedDict is one — a distinction only a type checker draws."""
    return [cast("dict[str, Any]", group) for group in groups]


def task(weight: float = 1.0) -> dict[str, Task]:
    return {"species": Classification("species", TargetInfo(classes=CLASSES), weight=weight)}


def module(
    tasks: Mapping[str, Task] | None = None,
    *,
    measured: bool = True,
    also: Mapping[str, type[torchmetrics.Metric]] | None = None,
    scheduler: Any = None,
    optimizer: Any = None,
    **learner_options: Any,
) -> TrainingModule:
    declared = task() if tasks is None else tasks
    metrics = {name: build_metrics(METRICS, one.facts()) for name, one in declared.items() if measured}
    if also:
        for collection in metrics.values():
            collection.add_metrics({label: metric() for label, metric in also.items()})
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

    def test_a_reading_that_means_a_image_reaches_the_tracker_and_the_numbers_reach_the_log(self) -> None:
        recorder = Recorder()
        fit = trainer(logger=recorder)

        fit.fit(module(), train_dataloaders=loader(RIGHT))

        titles = [title for title, _, _ in recorder.drawn]
        assert titles == ["train/species/confusion_matrix"]
        assert isinstance(recorder.drawn[0][1], Matrix)
        assert "train/species/accuracy" in recorder.scalars
        assert "train/species/confusion_matrix" not in recorder.scalars

    def test_a_sanity_check_draws_nothing_and_reports_nothing(self) -> None:
        """Lightning suppresses `self.log` during one; an image handed straight to a tracker needs the same."""
        recorder = Recorder()
        fit = trainer(logger=recorder, num_sanity_val_steps=1)

        fit.fit(module(), train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        drawn = sorted(title for title, _, _ in recorder.drawn)
        assert drawn == ["train/species/confusion_matrix", "val/species/confusion_matrix"], (
            "one image per stage that ran an epoch, and none for the batches the check looked at"
        )


class TestBatchTransforms:
    """The one place a batch is rewritten, and the only stage it is rewritten in."""

    @staticmethod
    def doubled(given: Batch) -> Batch:
        twice = {name: require_tensor(value, name=name) * 2 for name, value in given.inputs.items()}
        return Batch(inputs=twice, count=given.count)

    def test_a_training_batch_is_rewritten_by_what_was_installed(self) -> None:
        under_test = module()
        under_test.transform_batches(self.doubled)

        rewritten = under_test.on_after_batch_transfer(batch(RIGHT), 0)

        assert torch.equal(require_tensor(rewritten.inputs["species"], name="species"), RIGHT * 2)

    def test_evaluation_reads_the_data_as_it_is(self) -> None:
        """A report is about the data a run will be judged on, not about an image made up for training."""
        under_test = module()
        under_test.transform_batches(self.doubled)
        under_test.eval()

        rewritten = under_test.on_after_batch_transfer(batch(RIGHT), 0)

        assert torch.equal(require_tensor(rewritten.inputs["species"], name="species"), RIGHT)

    def test_a_run_that_installs_nothing_is_handed_the_very_batch_it_was_given(self) -> None:
        under_test, given = module(), batch(RIGHT)

        assert under_test.on_after_batch_transfer(given, 0) is given

    def test_what_was_installed_can_be_taken_back_out(self) -> None:
        under_test = module()
        under_test.transform_batches(self.doubled)

        under_test.transform_batches(None)

        rewritten = under_test.on_after_batch_transfer(batch(RIGHT), 0)
        assert torch.equal(require_tensor(rewritten.inputs["species"], name="species"), RIGHT)


class TestDirections:
    def test_it_says_which_way_each_measurement_is_better(self) -> None:
        """A table shows a best, and a direction guessed from a name is how a run shows the wrong one."""
        assert module().metric_directions() == {"species/accuracy": True, "species/confusion_matrix": None}

    def test_a_measurement_with_no_better_direction_says_so_rather_than_being_left_out(self) -> None:
        """Left out, it would be indistinguishable from a loss, which is the one thing assumed."""
        assert module().metric_directions()["species/confusion_matrix"] is None

    def test_a_run_that_measures_nothing_declares_nothing(self) -> None:
        assert module(measured=False).metric_directions() == {}


class TestStages:
    """A measurement is not always meaningful in every stage, and the one that is not says so."""

    def test_a_reading_declared_for_some_stages_is_not_taken_in_the_others(self) -> None:
        """Ranking against a gallery the encoder is still moving measures the drift, not the model.

        Left in, a run reports a number under a name that means something else in that column, and a
        reader comparing the three columns is comparing two different questions.
        """
        under_test = module(also={"still": WhenStill})
        fit = trainer()
        fit.fit(under_test, train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        assert "val/species/still" in fit.logged_metrics
        assert "train/species/still" not in fit.logged_metrics

    def test_what_a_stage_does_read_is_untouched_by_what_it_does_not(self) -> None:
        """The rest of the table is the point: one reading stepping aside takes none of the others."""
        under_test = module(also={"still": WhenStill})
        fit = trainer()
        fit.fit(under_test, train_dataloaders=loader(RIGHT), val_dataloaders=loader(RIGHT))

        assert fit.logged_metrics["train/species/accuracy"] == 1.0
        assert fit.logged_metrics["val/species/accuracy"] == 1.0

    def test_a_reading_absent_from_training_still_declares_which_way_it_is_better(self) -> None:
        """Directions were read off the training stage alone, so a reading missing there had none.

        A monitor that cannot find a direction takes the one it assumes, and `mode="min"` over a
        recall is a run that keeps its worst epoch and says nothing.
        """
        assert module(also={"still": WhenStill}).metric_directions()["species/still"] is True


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


class TestPreviews:
    """What a step produced, offered to whatever draws it, at the one moment the values exist."""

    @staticmethod
    def watching() -> tuple[list[StepPreview], Any]:
        seen: list[StepPreview] = []
        return seen, seen.append

    def test_every_step_is_offered_to_whoever_asked_to_see_one(self) -> None:
        seen, watcher = self.watching()
        under_test = module()
        under_test.preview_steps(watcher)

        trainer().fit(under_test, train_dataloaders=loader(RIGHT, WRONG), val_dataloaders=loader(RIGHT))

        assert [(one.stage, one.batch_index) for one in seen] == [
            (Stage.TRAIN, 0),
            (Stage.TRAIN, 1),
            (Stage.VAL, 0),
        ]

    def test_a_preview_carries_the_batch_the_step_read_and_what_it_produced(self) -> None:
        """Both halves, because a page draws the prediction over the very image that made it."""
        seen, watcher = self.watching()
        under_test = module()
        under_test.preview_steps(watcher)

        trainer(limit_train_batches=1, limit_val_batches=0).fit(under_test, train_dataloaders=loader(RIGHT))

        (preview,) = seen
        assert torch.equal(require_tensor(preview.batch.inputs["species"], name="species"), RIGHT)
        assert set(preview.output.predictions) == {"species"}
        assert preview.output.loss is not None

    def test_asking_twice_does_not_draw_twice(self) -> None:
        """Lightning calls a callback's setup once per stage, and a display must not double on it."""
        seen, watcher = self.watching()
        under_test = module()
        under_test.preview_steps(watcher)
        under_test.preview_steps(watcher)

        trainer(limit_train_batches=1, limit_val_batches=0).fit(under_test, train_dataloaders=loader(RIGHT))

        assert len(seen) == 1

    def test_a_module_nobody_watches_runs_as_it_did(self) -> None:
        fit = trainer(limit_val_batches=0)

        fit.fit(module(), train_dataloaders=loader(RIGHT))

        assert "train/loss" in fit.logged_metrics


def test_a_module_that_holds_no_learner_answers_for_that_itself() -> None:
    """The walk to the model is this module's own layout; a declaration's reader never wrote that path.

    Named the other way round, a refusal would tell a callback its declaration cannot find a path the
    callback did not write — and the path under the model, which is what a declaration does write, is
    the one the reader is answerable for.
    """
    with pytest.raises(LookupError, match="LightningModule cannot find"):
        module_at(L.LightningModule(), "backbone", reader="Freeze")
