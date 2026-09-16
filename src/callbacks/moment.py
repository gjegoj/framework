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
    """A point in a run as a declaration writes it: a share of the run, a whole epoch index, or neither.

    One grammar for every knob that names one — ``until``, ``over``, ``after`` — so a reader learns it
    once and a callback validates nothing of its own. Neither is how the run's own end is written: not
    as the number that would mean it, because that number is 1 and 1 is the one value the two readings
    disagree about. What it is counted in is the caller's business: the same declaration answers in
    epochs for a callback that acts once a year and in steps for one that acts every batch.

    Parameters:
        declared: A share of the run below 1, or a whole epoch index above it; left out, the moment is
            the run's own end — or its beginning, for a knob that opens.
        knob: What the config calls this, so a refusal names the line that has to change.
        opens: Whether it names where something begins rather than where it ends. A knob that closes
            refuses its empty case: an end at 0 means nothing ever happens.
    """

    declared: float | None
    knob: str
    opens: bool = False

    def __post_init__(self) -> None:
        """Refuse the one number this grammar cannot read, and the ones it cannot read at all.

        Every value but 1 either means the same under both readings or belongs to one of them alone: 0
        is where nothing happens on either count, 0.3 is no epoch, 3 is no share. One is the whole run
        *and* the first epoch, and the heuristic that tells the two apart — a whole number above 1 is
        an epoch — put it silently on the side of the share. A run declaring ``until: 1`` to hold its
        backbone for one epoch held it for all of them, and said ``Frozen until epoch 10 (step 400)``
        while it did, which reads like a declaration honoured.
        """
        if self.declared is None:
            return
        if self.declared == 1:
            raise ValueError(
                f"{self.knob!r} reads 1 as the whole run and as its first epoch, and it cannot be both: "
                f"leave {self.knob} out for the whole run, write 2 for the second epoch, or a share below 1."
            )
        whole = self.declared > 1 and self.declared == int(self.declared)
        empty = self.declared == 0 and not self.opens
        if not (0 <= self.declared < 1 or whole) or empty:
            span = "[0, 1)" if self.opens else "(0, 1)"
            raise ValueError(
                f"{self.knob!r} is a share of the run in {span} or a whole epoch index above 1, got {self.declared}."
            )

    @property
    def _number(self) -> float:
        """What both readings below count with: a knob left out is the end of the run, or its start."""
        if self.declared is not None:
            return self.declared
        return 0.0 if self.opens else 1.0

    def in_epochs(self, profile: FitProfile) -> Boundary:
        """The epoch this names, and the step that epoch starts at.

        A share is a point on the run's axis and the boundary is the first whole epoch at or past it,
        so a third of ten epochs is epoch 4 and all of it is the run's end. Measured before choosing
        ``ceil``: Python's ``round`` is banker's — ``round(2.5) == 2`` — so a quarter of a ten-epoch
        run would land one epoch short of its own share while a third landed past it.
        """
        declared = self._number
        epoch = int(declared) if declared > 1 else math.ceil(declared * profile.epochs)
        return Boundary(epoch, epoch * profile.steps_per_epoch)

    def in_steps(self, profile: FitProfile) -> Boundary:
        """The optimizer step this names, and the epoch that step falls in.

        Read against the run's steps rather than its epochs, so a callback acting every batch keeps
        the resolution it counts in; a whole epoch index is that epoch's first step.
        """
        declared = self._number
        step = int(declared) * profile.steps_per_epoch if declared > 1 else math.ceil(declared * profile.total_steps)
        return Boundary(step // profile.steps_per_epoch, step)
