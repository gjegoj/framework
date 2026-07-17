"""EMA-aware ModelCheckpoint — weights-only checkpoints store the averaged weights.

Lightning's ``dump_checkpoint`` invokes callbacks' ``on_save_checkpoint`` only when
``weights_only=False`` (a leftover gate from the pre-1.8 return-state contract). With
``save_weights_only=True`` the ``EmaCallback`` therefore never writes the averaged
weights into the checkpoint: the file stores the live training weights while the
monitored metric came from the averaged ones — a "best" checkpoint holding the wrong
model. This subclass re-applies the EMA hook on the weights-only path and defers to
the parent everywhere else (full checkpoints / no EMA in the trainer).
"""

from __future__ import annotations

from weakref import proxy

import lightning as L
from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint
from lightning.pytorch.profilers import Profiler


class EmaModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` whose weights-only checkpoints hold the EMA weights.

    Warmup guarding and averaged-model access stay ``EmaCallback``'s responsibility:
    its regular ``on_save_checkpoint`` hook is applied to the dumped checkpoint (it
    reads the averaged model, not the live module, so the validation swap state at
    save time is irrelevant). Without an ``EmaCallback`` on the trainer, or with
    ``save_weights_only=False``, behaviour is exactly the parent's.
    """

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        ema = self._find_ema(trainer)
        if not self.save_weights_only or ema is None:
            super()._save_checkpoint(trainer, filepath)
            return
        # Mirrors trainer.save_checkpoint (incl. its profiler block) + the parent's
        # bookkeeping, with the one hook Lightning skips for weights-only checkpoints
        # applied in between. ``profiler`` is assigned dynamically in Trainer.__init__,
        # so it is invisible to mypy (same as ``callbacks``).
        profiler: Profiler = getattr(trainer, "profiler")
        with profiler.profile("save_checkpoint"):
            checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)  # noqa: SLF001
            ema.on_save_checkpoint(trainer, trainer.lightning_module, checkpoint)
            trainer.strategy.save_checkpoint(checkpoint, filepath)
            trainer.strategy.barrier("Trainer.save_checkpoint")
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    @staticmethod
    def _find_ema(trainer: L.Trainer) -> EMAWeightAveraging | None:
        """Return the trainer's EMA callback, or ``None`` when EMA is not enabled.

        Matched against Lightning's base ``EMAWeightAveraging`` (which our ``EmaCallback``
        subclasses) — a run configured with the stock callback must get the same
        weights-only consistency fix, not silently fall back to live weights.
        """
        # Lightning assigns ``callbacks`` dynamically in __init__, so it is invisible to mypy.
        callbacks: list[L.Callback] = getattr(trainer, "callbacks", [])
        return next((callback for callback in callbacks if isinstance(callback, EMAWeightAveraging)), None)
