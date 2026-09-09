"""``FitProfile``: the facts of a fit, read off the trainer once and derived from there."""

from __future__ import annotations

from typing import Any

import lightning as L

from src.training.optim import FitProfile


class Estimating(L.Trainer):
    """A trainer stub answering only what a profile is derived from."""

    def __init__(self, max_epochs: int | None, stepping_batches: int) -> None:
        self._max_epochs = max_epochs
        self._stepping = stepping_batches

    @property
    def max_epochs(self) -> int | None:
        return self._max_epochs

    @property
    def estimated_stepping_batches(self) -> Any:
        return self._stepping


def test_the_profile_is_read_off_the_trainer() -> None:
    """``estimated_stepping_batches`` is the one number that already accounts for accumulation and ``drop_last``."""
    profile = FitProfile.of(Estimating(max_epochs=3, stepping_batches=18))

    assert profile == FitProfile(total_steps=18, epochs=3)
    assert profile.steps_per_epoch == 6


def test_a_trainer_that_declares_no_epochs_still_yields_a_profile() -> None:
    """``max_epochs=None`` reaches here from a trainer built for a single validation pass."""
    assert FitProfile.of(Estimating(max_epochs=None, stepping_batches=0)).epochs == 1
