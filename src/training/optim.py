"""Optimization building blocks: what a run needs to optimize and to schedule."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.utilities.types import LRSchedulerConfigType
    from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class FitProfile:
    """Facts about a run that only exist once fitting starts.

    The counterpart of ``DatasetFacts`` (facts inferred from data): these are
    inferred from the fit loop — dataset size, batch size and epoch count
    together — and only the training module can read them, because only it
    holds the trainer.

    Two fields, not three: the per-epoch step count follows from them, so it
    is derived rather than stored — an incoherent triple stays unrepresentable.
    """

    total_steps: int
    epochs: int

    @classmethod
    def of(cls, trainer: L.Trainer) -> FitProfile:
        """Read off the trainer once the loops are set up.

        Everything derives from ``estimated_stepping_batches``: it is the one value
        Lightning guarantees at that point (``num_training_batches`` is still infinite
        until then), and it already accounts for gradient accumulation, ``drop_last`` and
        any ``limit_*_batches``. A trainer without ``max_epochs`` — one built for a single
        validation pass — reads as one epoch, so a boundary can still be named.
        """
        return cls(total_steps=int(trainer.estimated_stepping_batches), epochs=max(int(trainer.max_epochs or 0), 1))

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps in one epoch; at least one, however short the loop."""
        return max(self.total_steps // self.epochs, 1)


type OptimizerFactory = Callable[[list[dict[str, Any]]], Optimizer]
"""Builds an optimizer over named parameter groups.

A *factory* rather than the object itself, because an optimizer needs the model's
parameters, which do not exist while config is being read.

``partial(torch.optim.AdamW, lr=1e-3)`` satisfies it: every torch constructor takes
group dicts, and a group naming no rate of its own inherits the factory's. Groups rather
than flat parameters even when no rate is overridden, because the groups are also what a
learning-rate monitor draws one line per.
"""

type SchedulerFactory = Callable[[Optimizer, FitProfile], LRSchedulerConfigType]
"""Builds a scheduler plus its Lightning scheduling policy, given fit-time facts.

A factory for the same reason as ``OptimizerFactory``, one step later: a schedule needs
both the built optimizer and how many steps the fit loop will actually run.
"""
