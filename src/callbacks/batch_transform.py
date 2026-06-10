"""Callback that applies a ``BatchTransform`` to training batches on a schedule.

The callback is a thin Observer: its only jobs are *when* (an optional cutoff
after a fraction of training) and *where* (the ``on_train_batch_start`` hook).
The actual mixing/stitching lives in the injected ``BatchTransform``. Transforms
are disabled during validation/test automatically — the hook fires only in
training.
"""

from __future__ import annotations

import logging

import lightning as L

from src.core.entities import Batch
from src.core.ports import BatchTransform

log = logging.getLogger(__name__)


class BatchTransformCallback(L.Callback):
    """Applies a batch transform to each training batch until a cutoff epoch.

    Parameters:
        transform (BatchTransform): The cross-sample transform to apply.
        disable_after_fraction (float): Fraction of ``max_epochs`` after which the
            transform stops being applied. ``1.0`` → active for all of training
            (common for MixUp); ``0.5`` → off for the second half (lets the model
            fine-tune on clean data). Must be in ``(0, 1]``.
    """

    def __init__(self, transform: BatchTransform, disable_after_fraction: float = 1.0) -> None:
        super().__init__()
        if not (0.0 < disable_after_fraction <= 1.0):
            raise ValueError(
                f"BatchTransformCallback: disable_after_fraction must be in (0, 1], got {disable_after_fraction}."
            )
        self._transform = transform
        self._disable_after_fraction = disable_after_fraction
        self._disable_epoch: int | None = None
        self._disabled_logged = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        max_epochs = trainer.max_epochs or 0
        self._disable_epoch = int(self._disable_after_fraction * max_epochs) if max_epochs > 0 else None
        until = "all epochs" if self._disable_epoch is None else f"epoch {self._disable_epoch}"
        log.info("Batch transform %s active until %s.", type(self._transform).__name__, until)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not self._is_active(trainer.current_epoch):
            if not self._disabled_logged:
                self._disabled_logged = True
                log.info(
                    "Batch transform %s disabled after epoch %s.",
                    type(self._transform).__name__,
                    self._disable_epoch,
                )
            return
        if not isinstance(batch, Batch):
            return
        result = self._transform(batch)
        batch.inputs.update(result.inputs)
        batch.targets.update(result.targets)

    def _is_active(self, current_epoch: int) -> bool:
        return self._disable_epoch is None or current_epoch < self._disable_epoch
