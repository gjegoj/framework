"""``ApplyBatchTransform``: where a batch transform runs, and until when."""

from __future__ import annotations

from typing import cast

import lightning as L
import pytest
import torch

from src.callbacks import ApplyBatchTransform
from src.callbacks.registry import callback_registry
from src.core import Batch
from tests.support.lightning import quiet_trainer


class Doubling:
    """A stand-in transform whose effect is unmistakable.

    Inherits nothing: ``BatchTransform`` is a callable contract, so satisfying it
    is a matter of having the right ``__call__``.
    """

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


def apply(callback: ApplyBatchTransform, given: Batch, epoch: int = 0) -> Batch:
    callback.on_train_batch_start(trainer_at(epoch), cast("L.LightningModule", None), given, 0)
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
