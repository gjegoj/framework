"""``FitProfile``: the facts that exist only once fitting starts."""

from __future__ import annotations

from src.training import FitProfile


def test_the_per_epoch_step_count_follows_from_the_run() -> None:
    """Two fields carry a coherent third; an incoherent triple stays unrepresentable."""
    assert FitProfile(total_steps=100, epochs=10).steps_per_epoch == 10


def test_a_run_shorter_than_its_epoch_count_still_steps_once() -> None:
    """Integer division would report zero steps an epoch, which no schedule can size against."""
    assert FitProfile(total_steps=3, epochs=10).steps_per_epoch == 1
