"""EMA callback — keeps a shadow copy of model weights using PyTorch's AveragedModel.

EMA weights are swapped into the model for the full validation pass so that
``ModelCheckpoint`` (which fires in ``on_validation_epoch_end``) saves EMA
weights automatically — no custom ``ModelCheckpoint`` subclass needed.

Hook order for Lightning callbacks (in registration order):
  on_validation_start      → EmaCallback swaps training → EMA weights
  on_validation_epoch_end  → ModelCheckpoint saves (EMA weights active)
  on_validation_end        → EmaCallback restores training weights
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import lightning as L
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class EmaCallback(L.Callback):
    """Maintains an Exponential Moving Average of model weights.

    Parameters:
        decay (float): EMA decay factor; typical range 0.99–0.9999.
        warmup_fraction (float): Fraction of total training steps before EMA
            updates begin. 0.0 → start immediately.
        use_buffers (bool): Also average BatchNorm running statistics.
    """

    def __init__(
        self,
        decay: float = 0.999,
        warmup_fraction: float = 0.0,
        use_buffers: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EmaCallback: decay must be in (0, 1), got {decay}.")
        if not (0.0 <= warmup_fraction < 1.0):
            raise ValueError(f"EmaCallback: warmup_fraction must be in [0, 1), got {warmup_fraction}.")
        self._decay = decay
        self._warmup_fraction = warmup_fraction
        self._use_buffers = use_buffers

        self._ema_model: AveragedModel | None = None
        self._training_backup: dict[str, Any] | None = None
        self._start_step: int = 0

    # ---------------------------------------------------------------- setup

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._ema_model = AveragedModel(
            pl_module,
            multi_avg_fn=get_ema_multi_avg_fn(self._decay),
            use_buffers=self._use_buffers,
        ).to(pl_module.device)
        self._start_step = int(self._warmup_fraction * trainer.estimated_stepping_batches)
        log.info("EMA initialised — decay=%.4f, warmup until step %d.", self._decay, self._start_step)

    # -------------------------------------------------------------- update

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._ema_model is None or trainer.global_step < self._start_step:
            return
        self._ema_model.update_parameters(pl_module)

    # -------------------------------------------------- validation swap

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._ema_model is None or trainer.global_step < self._start_step:
            return
        self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._restore_training_weights(pl_module)

    # ------------------------------------------------------- test swap

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._ema_model is None:
            return
        self._swap_to_ema(pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._restore_training_weights(pl_module)

    # ---------------------------------------------------------------- helpers

    def _swap_to_ema(self, pl_module: L.LightningModule) -> None:
        self._training_backup = deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self._ema_model.module.state_dict(), strict=False)  # type: ignore[union-attr]

    def _restore_training_weights(self, pl_module: L.LightningModule) -> None:
        if self._training_backup is None:
            return
        pl_module.load_state_dict(self._training_backup, strict=False)
        self._training_backup = None
