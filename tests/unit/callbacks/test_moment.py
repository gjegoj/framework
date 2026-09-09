"""``moment``: one instant of a run, named in both currencies a reader can act on."""

from __future__ import annotations

from typing import Any

import lightning as L
import pytest

from src.callbacks.moment import at_epoch, at_step, declared_moment, epoch_at, step_at
from src.training.optim import FitProfile


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
        assert at_step(trainer, epoch * FitProfile.of(trainer).steps_per_epoch) == at_epoch(trainer, epoch)


@pytest.mark.parametrize(("epochs", "batches"), [(0, 10), (3, 0), (0, 0)])
def test_a_run_that_declares_nothing_to_divide_by_still_prints_a_moment(epochs: int, batches: int) -> None:
    """A boundary is a log line; it must not be the thing that ends the run.

    `max_epochs=None` and an unset estimate both reach here — from a trainer built
    for a single validation pass, or before the loops know their lengths.
    """
    assert at_epoch(Estimating(epochs, batches), 1).startswith("epoch 1 (step ")


@pytest.mark.parametrize(
    ("declared", "epochs", "epoch"),
    [(0.25, 10, 3), (0.34, 10, 4), (0.5, 10, 5), (1.0, 10, 10), (3, 10, 3), (3.0, 10, 3)],
)
def test_a_share_names_the_first_epoch_at_or_past_it_and_a_whole_number_names_itself(
    declared: float, epochs: int, epoch: int
) -> None:
    """One grammar for every knob that names a moment: ``0.34`` of ten epochs is 3.4 epochs in, so the
    boundary is epoch 4; ``3`` is epoch 3 whatever the run's length."""
    assert epoch_at(declared, epochs) == epoch


def test_every_share_reads_the_same_way_whichever_side_of_a_half_it_falls() -> None:
    """Measured before this existed: ``round`` is banker's, so ``0.25`` and ``0.35`` of ten epochs came out
    as 2 and 4 — one short of the share, one past it."""
    assert (epoch_at(0.25, 10), epoch_at(0.35, 10)) == (3, 4)


def test_a_step_is_the_first_at_or_past_the_share_or_the_first_of_the_named_epoch() -> None:
    trainer = Estimating(max_epochs=4, stepping_batches=100)  # 25 steps an epoch

    assert step_at(0.1, trainer) == 10
    assert step_at(0.105, trainer) == 11  # 10.5 steps in: the first whole step past it
    assert step_at(2, trainer) == 50


@pytest.mark.parametrize(
    ("value", "role"),
    [(0, "end"), (1, "start"), (2.5, "end"), (2.5, "start"), (-0.5, "start"), (1.5, "end")],
)
def test_a_moment_outside_the_run_is_refused_naming_the_knob(value: float, role: str) -> None:
    """An end at 0 never happens, a start at 1 never comes, and a fraction above 1 is neither a share nor
    an epoch — each refused at construction, naming the knob the way the callback spelled it."""
    with pytest.raises(ValueError, match="knob"):
        declared_moment(value, owner="Owner", knob="knob", role=role)  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "role"), [(0, "start"), (1, "end"), (0.5, "start"), (2, "start"), (7, "end")])
def test_a_moment_inside_the_run_is_returned_as_declared(value: float, role: str) -> None:
    assert declared_moment(value, owner="Owner", knob="knob", role=role) == value  # type: ignore[arg-type]
