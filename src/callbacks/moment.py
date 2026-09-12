"""Where in a run something begins or ends: declared the same way, and said the same way, by every callback."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.training import FitProfile


@dataclass(frozen=True, slots=True)
class Boundary:
    """A point in the run, counted both ways a reader can act on.

    A config declares a *share* of the run, a callback acts on an *epoch*, and a tracker counts in
    *steps*; steps-per-epoch follows from gradient accumulation and a dropped tail, so neither number
    can be worked out from the other by whoever reads the line.
    """

    epoch: int
    step: int

    def __str__(self) -> str:
        return f"epoch {self.epoch} (step {self.step})"


@dataclass(frozen=True, slots=True)
class Moment:
    """A point in a run as a declaration writes it: a share of the run, or a whole epoch index.

    One grammar for every knob that names one — ``until``, ``over``, ``after`` — so a reader learns it
    once and a callback validates nothing of its own. What it is counted in is the caller's business:
    the same declaration answers in epochs for a callback that acts once a year and in steps for one
    that acts every batch.

    Parameters:
        declared: A share of the run in ``[0, 1]``, or a whole epoch index above 1.
        knob: What the config calls this, so a refusal names the line that has to change.
        opens: Whether it names where something begins rather than where it ends. Each refuses its own
            empty case: an end at 0 means nothing ever happens, a beginning at 1 never comes.
    """

    declared: float
    knob: str
    opens: bool = False

    def __post_init__(self) -> None:
        whole = self.declared > 1 and self.declared == int(self.declared)
        empty = self.declared == (1 if self.opens else 0)
        if not (0 <= self.declared <= 1 or whole) or empty:
            span = "[0, 1)" if self.opens else "(0, 1]"
            raise ValueError(
                f"{self.knob!r} is a share of the run in {span} or a whole epoch index, got {self.declared}."
            )

    def in_epochs(self, profile: FitProfile) -> Boundary:
        """The epoch this names, and the step that epoch starts at.

        A share is a point on the run's axis and the boundary is the first whole epoch at or past it,
        so a third of ten epochs is epoch 4 and all of it is the run's end. Measured before choosing
        ``ceil``: Python's ``round`` is banker's — ``round(2.5) == 2`` — so a quarter of a ten-epoch
        run would land one epoch short of its own share while a third landed past it.
        """
        epoch = int(self.declared) if self.declared > 1 else math.ceil(self.declared * profile.epochs)
        return Boundary(epoch, epoch * profile.steps_per_epoch)

    def in_steps(self, profile: FitProfile) -> Boundary:
        """The optimizer step this names, and the epoch that step falls in.

        Read against the run's steps rather than its epochs, so a callback acting every batch keeps
        the resolution it counts in; a whole epoch index is that epoch's first step.
        """
        step = (
            int(self.declared) * profile.steps_per_epoch
            if self.declared > 1
            else math.ceil(self.declared * profile.total_steps)
        )
        return Boundary(step // profile.steps_per_epoch, step)
