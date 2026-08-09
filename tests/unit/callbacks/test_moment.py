"""``moment``: one instant of a run, named in both currencies a reader can act on."""

from __future__ import annotations

from typing import Any

import lightning as L
import pytest

from src.callbacks.moment import at_epoch, at_step, steps_per_epoch


class Estimating(L.Trainer):
    """A trainer stub answering only what a moment is derived from."""

    def __init__(self, max_epochs: int, stepping_batches: int) -> None:
        self._max_epochs = max_epochs
        self._stepping = stepping_batches

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def estimated_stepping_batches(self) -> Any:
        return self._stepping


def test_a_boundary_is_named_in_epochs_and_in_steps_at_once() -> None:
    """Three currencies describe one instant, and each reader wants a different one.

    Config declares a share, a callback acts on an epoch, and a tracker's x-axis
    counts steps. A line naming one leaves the reader converting — and getting it
    wrong, because steps-per-epoch follows gradient accumulation and `drop_last`,
    not arithmetic done in the head.
    """
    trainer = Estimating(max_epochs=3, stepping_batches=18)

    assert at_epoch(trainer, 2) == "epoch 2 (step 12)"
    assert at_step(trainer, 12) == "epoch 2 (step 12)"


def test_the_two_readings_of_one_instant_agree() -> None:
    """Whichever currency a callback happens to hold, the sentence must come out the same."""
    trainer = Estimating(max_epochs=4, stepping_batches=100)

    for epoch in range(4):
        assert at_step(trainer, epoch * steps_per_epoch(trainer)) == at_epoch(trainer, epoch)


@pytest.mark.parametrize(("epochs", "batches"), [(0, 10), (3, 0), (0, 0)])
def test_a_run_that_declares_nothing_to_divide_by_still_prints_a_moment(epochs: int, batches: int) -> None:
    """A boundary is a log line; it must not be the thing that ends the run.

    `max_epochs=None` and an unset estimate both reach here — from a trainer built
    for a single validation pass, or before the loops know their lengths.
    """
    assert at_epoch(Estimating(epochs, batches), 1).startswith("epoch 1 (step ")
