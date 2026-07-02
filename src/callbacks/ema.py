"""EMA callback — a thin facade over Lightning's ``EMAWeightAveraging``.

Lightning's built-in already handles the hard parts correctly: EMA weights are
persisted into checkpoints (and restored on resume) via
``on_save_checkpoint``/``on_load_checkpoint`` and copied into the model at
``on_train_end``. We add two things on top:

* ``warmup_fraction`` expresses the update-start point as a fraction of total
  training steps (resolution-independent of the schedule, which isn't known at
  config time), resolved to Lightning's absolute ``update_starting_at_step`` once
  the step count is known at ``setup``.
* A guard so the averaged weights are only swapped in for validation, written
  into checkpoints, and copied into the model at train end **after averaging has
  actually begun**. Until the first EMA update the averaged model is a frozen copy
  of the initial (random) weights; Lightning uses it unconditionally, which makes
  validation during warmup evaluate the untrained model (flat, meaningless val
  loss) and — worse — makes every checkpoint saved during warmup store those
  initial weights as its ``state_dict`` while its monitored metric came from the
  live weights (a "best" checkpoint holding random weights). With the guard,
  warmup validation and checkpoints use the live training weights; the EMA weights
  take over once averaging starts.
"""

from __future__ import annotations

import logging
from typing import Any, override

import lightning as L
from lightning.pytorch.callbacks import EMAWeightAveraging

log = logging.getLogger(__name__)


class EmaCallback(EMAWeightAveraging):
    """Exponential Moving Average of model weights with fractional warmup.

    Parameters:
        decay (float): EMA decay factor; typical range 0.99–0.9999.
        warmup_fraction (float): Fraction of total training steps before EMA
            updates begin. 0.0 → start immediately.
        use_buffers (bool): Also average BatchNorm running statistics.
        **kwargs: Forwarded to ``EMAWeightAveraging`` (e.g. ``device``,
            ``update_every_n_steps``).
    """

    def __init__(
        self,
        decay: float = 0.999,
        warmup_fraction: float = 0.0,
        use_buffers: bool = True,
        **kwargs: Any,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EmaCallback: decay must be in (0, 1), got {decay}.")
        if not (0.0 <= warmup_fraction < 1.0):
            raise ValueError(f"EmaCallback: warmup_fraction must be in [0, 1), got {warmup_fraction}.")
        super().__init__(decay=decay, use_buffers=use_buffers, **kwargs)
        self._decay = decay  # parent discards it into avg_fn; keep for logging
        self._warmup_fraction = warmup_fraction

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Create the averaged model and resolve fractional warmup to a step."""
        super().setup(trainer, pl_module, stage)
        if stage == "fit":
            self.update_starting_at_step = int(self._warmup_fraction * trainer.estimated_stepping_batches)
            log.info(
                "EMA initialized: decay=%s, updates start at step %d.",
                self._decay,
                self.update_starting_at_step,
            )

    def _averaging_has_started(self) -> bool:
        """Whether at least one EMA update has run — before that the averaged model is just init weights.

        ``_latest_update_step`` (owned by the parent) starts at ``0`` and is set to the global step of
        the first averaging update, so ``> 0`` marks the moment the averaged model becomes meaningful.
        """
        return self._latest_update_step > 0

    @override
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Swap the averaged weights in for validation only once averaging has started (else use live weights)."""
        if self._averaging_has_started():
            super().on_validation_epoch_start(trainer, pl_module)

    @override
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Swap the live weights back after validation — kept symmetric with ``on_validation_epoch_start``."""
        if self._averaging_has_started():
            super().on_validation_epoch_end(trainer, pl_module)

    @override
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        """Write the averaged weights into the checkpoint only once averaging has started.

        The parent unconditionally replaces ``checkpoint["state_dict"]`` with the averaged model —
        during warmup that would persist the frozen initial weights under a checkpoint whose
        monitored metric was produced by the live weights. Until averaging starts, the checkpoint
        keeps the plain (live) ``state_dict``; the parent's ``on_load_checkpoint`` handles that
        format via its no-averaging-state branch, so resume works from either kind.
        """
        if self._averaging_has_started():
            super().on_save_checkpoint(trainer, pl_module, checkpoint)

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Copy the EMA weights into the model at the end — but only if averaging ever ran.

        Guards the degenerate case (warmup so long that averaging never started) where the parent would
        otherwise overwrite the trained model with the frozen initial weights.
        """
        if self._averaging_has_started():
            super().on_train_end(trainer, pl_module)
