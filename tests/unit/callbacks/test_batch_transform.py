"""``ApplyBatchTransform``: where a batch transform runs, and until when."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import partial
from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest
import torch

from src.callbacks import ApplyBatchTransform
from src.callbacks.registry import callback_registry
from src.core import Batch
from src.tasks import Task
from src.training import TrainingData, TrainingModule
from tests.support.entities import a_task
from tests.support.fakes import a_composite
from tests.support.lightning import quiet_trainer
from tests.support.tables import in_memory_pipeline


class Doubling:
    """A stand-in transform whose effect is unmistakable.

    Inherits nothing: ``BatchTransform`` is a structural contract — ``for_tasks`` binds it
    to the run's tasks and hands back the callable — so satisfying it is a matter of
    having the right two methods.
    """

    def __init__(self) -> None:
        self.bound_to: Sequence[Task] | None = None

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        self.bound_to = tuple(tasks)
        return self

    def __call__(self, batch: Batch) -> Batch:
        return Batch(
            inputs={name: value * 2 for name, value in batch.inputs.items()},
            targets={"mixed": torch.ones(1)},
            meta=batch.meta,
        )


def batch() -> Batch:
    return Batch(inputs={"image": torch.ones(2, 3)}, targets={"label": torch.zeros(2)})


def trainer_at(epoch: int, max_epochs: int = 10) -> L.Trainer:
    trainer = quiet_trainer(max_epochs=max_epochs)
    trainer.fit_loop.epoch_progress.current.completed = epoch  # what Trainer.current_epoch reads
    return trainer


def module_of(*tasks: Task) -> L.LightningModule:
    """What the callback reads at setup: the module's ``tasks``, nothing else."""
    return cast("L.LightningModule", SimpleNamespace(tasks=tasks))


def apply(callback: ApplyBatchTransform, given: Batch, epoch: int = 0) -> Batch:
    callback.setup(trainer_at(epoch), module_of(a_task()), stage="fit")
    callback.on_train_batch_start(trainer_at(epoch), module_of(a_task()), given, 0)
    return given


def test_the_batch_is_written_back_because_the_hook_cannot_replace_it() -> None:
    """Lightning discards what a callback returns, so the result has to be assigned in."""
    given = apply(ApplyBatchTransform(Doubling()), batch())

    assert torch.equal(given.inputs["image"], torch.full((2, 3), 2.0))


def test_the_write_back_is_exact_rather_than_a_merge() -> None:
    """A transform that changes which keys exist has to be represented faithfully."""
    given = apply(ApplyBatchTransform(Doubling()), batch())

    assert set(given.targets) == {"mixed"}


def test_it_is_silent_once_the_schedule_is_over() -> None:
    """Finishing on clean data is the point of a schedule."""
    given = apply(ApplyBatchTransform(Doubling(), until=0.5), batch(), epoch=7)

    assert torch.equal(given.inputs["image"], torch.ones(2, 3))


def test_it_is_active_before_the_cutoff() -> None:
    given = apply(ApplyBatchTransform(Doubling(), until=0.5), batch(), epoch=2)

    assert torch.equal(given.inputs["image"], torch.full((2, 3), 2.0))


def test_the_whole_run_is_the_default() -> None:
    given = apply(ApplyBatchTransform(Doubling()), batch(), epoch=9)

    assert torch.equal(given.inputs["image"], torch.full((2, 3), 2.0))


@pytest.mark.parametrize("until", [0.0, 1.5, -0.5])
def test_a_cutoff_outside_the_run_is_refused(until: float) -> None:
    with pytest.raises(ValueError, match="until"):
        ApplyBatchTransform(Doubling(), until=until)


def test_it_is_reachable_from_config_by_name() -> None:
    built = callback_registry.create("batch_transform", transform=Doubling())

    assert isinstance(built, ApplyBatchTransform)


def test_validation_batches_never_reach_it() -> None:
    """Not a flag: the one hook that reaches a batch fires in training only.

    Scoped to the batch hooks because the class also announces its window at fit
    start, which touches no batch at all — adding `on_validation_batch_start` still
    fails here, which is the property under test.

    ``hasattr`` cannot say this — Lightning's base class defines every hook as a
    no-op, so all of them are present on any callback.
    """
    reaching_batches = {name for name in vars(ApplyBatchTransform) if "_batch_" in name}
    assert reaching_batches == {"on_train_batch_start"}


def test_the_transform_is_bound_to_the_modules_tasks_when_the_trainer_sets_it_up() -> None:
    """The tasks are the module's to declare; the callback reads them there, the way any Lightning callback would."""
    doubling = Doubling()
    task = a_task(name="species")

    ApplyBatchTransform(doubling).setup(trainer_at(0), module_of(task), stage="fit")

    assert doubling.bound_to == (task,)


def test_a_batch_before_setup_is_refused_naming_setup() -> None:
    with pytest.raises(RuntimeError, match="setup"):
        ApplyBatchTransform(Doubling()).on_train_batch_start(trainer_at(0), module_of(), batch(), 0)


def test_a_module_without_tasks_is_refused_by_name() -> None:
    """A batch transform rewrites tasks' targets; a module declaring none has nothing for it to do."""
    with pytest.raises(ValueError, match="tasks"):
        ApplyBatchTransform(Doubling()).setup(trainer_at(0), cast("L.LightningModule", SimpleNamespace()), stage="fit")


class Recording:
    """A transform that changes nothing and counts the batches it was handed."""

    def __init__(self) -> None:
        self.applied = 0

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        return self

    def __call__(self, batch: Batch) -> Batch:
        self.applied += 1
        return batch


def test_under_a_fit_the_schedule_ends_where_it_was_announced(caplog: pytest.LogCaptureFixture) -> None:
    """Two epochs at ``until=0.5``: every batch of epoch 0 is transformed, epoch 1 is announced at the start and reported once it stops."""
    pipeline, _ = in_memory_pipeline()  # four training rows: two batches an epoch
    transform = Recording()
    module = TrainingModule(model=a_composite(2), tasks=[a_task()], optimizer_factory=partial(torch.optim.SGD, lr=0.1))
    trainer = quiet_trainer(max_epochs=2, callbacks=[ApplyBatchTransform(transform, until=0.5)], limit_val_batches=0)

    with caplog.at_level(logging.INFO):
        trainer.fit(module, datamodule=TrainingData(pipeline, batch_size=2))

    assert transform.applied == 2
    assert "Recording applied until epoch 1" in caplog.text
    assert "Recording stopped at epoch 1" in caplog.text


def test_a_whole_number_until_is_the_epoch_the_transform_stops_at() -> None:
    """The same grammar as ``Freeze``: ``until: 3`` stops at epoch 3, ``until: 0.3`` at a share of the run."""
    doubled = apply(ApplyBatchTransform(Doubling(), until=3), batch(), epoch=2)
    untouched = apply(ApplyBatchTransform(Doubling(), until=3), batch(), epoch=3)

    assert torch.equal(doubled.inputs["image"], torch.full((2, 3), 2.0))
    assert torch.equal(untouched.inputs["image"], torch.ones(2, 3))
