"""Running a batch transform on training batches, for as long as it is wanted."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, override

import lightning as L

from src.callbacks.moment import at_epoch
from src.core.entities import Batch

if TYPE_CHECKING:
    from src.core.ports import BatchTransform

log = logging.getLogger(__name__)


class ApplyBatchTransform(L.Callback):
    """Apply a batch transform to training batches until a point in the run.

    Validation and test are excluded by construction: the hook fires in training only.
    Lightning discards a callback's return value, so the result is assigned into the batch
    that was handed over, replacing its fields wholesale.

    Parameters:
        transform (BatchTransform): What to apply.
        until (float): Share of the run it stays active for; ``0.8`` stops for the last
            fifth, so a model finishes on the kind of data it is evaluated on.
    """

    def __init__(self, transform: BatchTransform, until: float = 1.0) -> None:
        super().__init__()
        if not 0.0 < until <= 1.0:
            raise ValueError(f"ApplyBatchTransform until must be a share of the run in (0, 1], got {until}.")
        self._transform = transform
        self._until = until
        self._reported = False

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Say when it will stop before it starts, rather than only once it has."""
        log.info(
            "%s applied until %s.",
            type(self._transform).__name__,
            at_epoch(trainer, self._stops_at(trainer)),
        )

    @override
    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if not isinstance(batch, Batch):
            return
        if not self._is_active(trainer):
            if not self._reported:
                self._reported = True
                log.info(
                    "%s stopped at %s.",
                    type(self._transform).__name__,
                    at_epoch(trainer, trainer.current_epoch),
                )
            return
        mixed = self._transform(batch)
        batch.inputs, batch.targets = mixed.inputs, mixed.targets

    def _is_active(self, trainer: L.Trainer) -> bool:
        max_epochs = trainer.max_epochs or 0
        return max_epochs <= 0 or trainer.current_epoch < self._until * max_epochs

    def _stops_at(self, trainer: L.Trainer) -> int:
        """The first epoch it no longer applies to — the same boundary ``_is_active`` reads."""
        max_epochs = int(trainer.max_epochs or 0)
        return max(0, math.ceil(self._until * max_epochs)) if max_epochs > 0 else 0
