"""EMA callback — a thin facade over Lightning's ``EMAWeightAveraging``.

Lightning's built-in already handles the hard parts correctly: EMA weights are
swapped in for validation, persisted into checkpoints (and restored on resume)
via ``on_save_checkpoint``/``on_load_checkpoint``, and copied into the model at
``on_train_end``. We add one ergonomic on top: ``warmup_fraction`` expresses the
update-start point as a fraction of total training steps (resolution-independent
of the schedule, which isn't known at config time), resolved to Lightning's
absolute ``update_starting_at_step`` once the step count is known at ``setup``.
"""

from __future__ import annotations

import logging
from typing import Any

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
