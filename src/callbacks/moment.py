"""Where in a run something begins or ends, said the same way by every callback."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning as L


def steps_per_epoch(trainer: L.Trainer) -> int:
    """Optimizer steps in one epoch, as the trainer estimates them.

    ``estimated_stepping_batches`` rather than the loader's length: it is the one
    number that already accounts for gradient accumulation, ``drop_last``, and any
    ``limit_*_batches``, and it is the same number the schedulers are built from.
    """
    epochs = max(int(trainer.max_epochs or 0), 1)
    return max(1, int(trainer.estimated_stepping_batches) // epochs)


def at_epoch(trainer: L.Trainer, epoch: int) -> str:
    """``epoch 2 (step 12)``, from the epoch a callback holds.

    Config declares a *share* of the run, a callback acts on an *epoch*, a tracker counts
    in *steps*; a boundary is announced in both of the two a reader can act on, because
    steps-per-epoch depends on gradient accumulation and ``drop_last``.
    """
    return f"epoch {epoch} (step {epoch * steps_per_epoch(trainer)})"


def at_step(trainer: L.Trainer, step: int) -> str:
    """``epoch 2 (step 12)``, from the step a callback holds."""
    return f"epoch {step // steps_per_epoch(trainer)} (step {step})"
