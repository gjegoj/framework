"""Where in a run something begins or ends: declared the same way and said the same way by every callback."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from src.training.optim import FitProfile

if TYPE_CHECKING:
    import lightning as L


def at_epoch(trainer: L.Trainer, epoch: int) -> str:
    """``epoch 2 (step 12)``, from the epoch a callback holds.

    Config declares a *share* of the run, a callback acts on an *epoch*, a tracker counts
    in *steps*; a boundary is announced in both of the two a reader can act on, because
    steps-per-epoch depends on gradient accumulation and ``drop_last``.
    """
    return f"epoch {epoch} (step {epoch * FitProfile.of(trainer).steps_per_epoch})"


def at_step(trainer: L.Trainer, step: int) -> str:
    """``epoch 2 (step 12)``, from the step a callback holds."""
    return f"epoch {step // FitProfile.of(trainer).steps_per_epoch} (step {step})"


type Role = Literal["start", "end"]
"""Whether the knob names where something begins (``after``) or where it ends (``until``, ``over``)."""


def declared_moment(value: float, *, owner: str, knob: str, role: Role) -> float:
    """Validate a knob that names a point in the run, and hand it back.

    One grammar for every such knob — ``until``, ``over``, ``after`` — so a reader learns it once:
    a share of the run in ``[0, 1]``, or a whole epoch index above 1. Each role refuses its own
    empty case: an end at 0 means nothing ever happens, a start at 1 never comes. A fraction
    above 1 is neither a share nor an epoch. Refused at construction, naming the knob as the
    callback spells it, so a bad declaration dies before the run and not at its first epoch.
    """
    whole = value > 1 and value == int(value)
    empty = value == (0 if role == "end" else 1)
    if not (0 <= value <= 1 or whole) or empty:
        span = "(0, 1]" if role == "end" else "[0, 1)"
        raise ValueError(f"{owner} {knob} is a share of the run in {span} or a whole epoch index, got {value}.")
    return value


def epoch_at(declared: float, max_epochs: int) -> int:
    """The epoch a declared moment names.

    A share is a point on the run's axis, and the boundary is the first whole epoch at or past
    it — ``ceil`` — so ``0.34`` of ten epochs is epoch 4 and ``1.0`` is the run's end; a whole
    number above 1 is that epoch itself. Measured before choosing: Python's ``round`` is
    banker's (``round(2.5) == 2``), so an annealing window of ``0.25`` over ten epochs came out
    as 2 while ``0.35`` came out as 4 — one short of its share, one past it. ``ceil`` reads every
    share the same way, and it is what the freeze and the batch transform already did.
    """
    if declared > 1:
        return int(declared)
    return math.ceil(declared * max_epochs)


def step_at(declared: float, trainer: L.Trainer) -> int:
    """The optimizer step a declared moment names, for a callback that counts in steps.

    A share is read against the run's steps rather than its epochs, so it keeps the
    resolution a step-wise callback has; a whole epoch index is that epoch's first step.
    """
    profile = FitProfile.of(trainer)
    if declared > 1:
        return int(declared) * profile.steps_per_epoch
    return math.ceil(declared * profile.total_steps)
