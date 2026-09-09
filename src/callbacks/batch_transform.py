"""Running a batch transform on training batches, for as long as it is wanted."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

import lightning as L

from src.callbacks.moment import at_epoch, declared_moment, epoch_at
from src.core.entities import Batch

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.transforms import BatchTransform

log = logging.getLogger(__name__)


class ApplyBatchTransform(L.Callback):
    """Apply a batch transform to training batches until a point in the run.

    Validation and test are excluded by construction: the hook fires in training only.
    Lightning discards a callback's return value, so the result is assigned into the batch
    that was handed over, replacing its fields wholesale.

    Bound to the run's tasks in ``setup``, read off the module (ADR-0004); a task the
    transform cannot serve is refused there, before the first batch.

    Parameters:
        transform (BatchTransform): What to apply, still unbound.
        until (float): Share of the run it stays active for; ``0.8`` stops for the last
            fifth, so a model finishes on the kind of data it is evaluated on.
    """

    def __init__(self, transform: BatchTransform, until: float = 1.0) -> None:
        super().__init__()
        self._declared = transform
        self._transform: Callable[[Batch], Batch] | None = None
        self._until = declared_moment(until, owner="ApplyBatchTransform", knob="until", role="end")
        self._reported = False

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        tasks = getattr(pl_module, "tasks", None)
        if tasks is None:
            raise ValueError(
                f"{type(self._declared).__name__} rewrites tasks' targets, but {type(pl_module).__name__} "
                f"declares no 'tasks'. TrainingModule publishes its tasks; a module of your own must too."
            )
        self._transform = self._declared.for_tasks(tasks)

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Say when it will stop before it starts, rather than only once it has."""
        log.info(
            "%s applied until %s.",
            type(self._declared).__name__,
            at_epoch(trainer, self._stops_at(trainer)),
        )

    @override
    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if not isinstance(batch, Batch):
            return
        if self._transform is None:
            raise RuntimeError(
                f"{type(self._declared).__name__} received a batch before setup bound it to the "
                f"run's tasks; Lightning runs setup first, so this callback was driven by hand."
            )
        if not self._is_active(trainer):
            if not self._reported:
                self._reported = True
                log.info(
                    "%s stopped at %s.",
                    type(self._declared).__name__,
                    at_epoch(trainer, trainer.current_epoch),
                )
            return
        mixed = self._transform(batch)
        batch.inputs, batch.targets = mixed.inputs, mixed.targets

    def _is_active(self, trainer: L.Trainer) -> bool:
        max_epochs = int(trainer.max_epochs or 0)
        return max_epochs <= 0 or trainer.current_epoch < self._stops_at(trainer)

    def _stops_at(self, trainer: L.Trainer) -> int:
        """The first epoch it no longer applies to — the same boundary ``_is_active`` reads."""
        max_epochs = int(trainer.max_epochs or 0)
        return epoch_at(self._until, max_epochs) if max_epochs > 0 else 0
