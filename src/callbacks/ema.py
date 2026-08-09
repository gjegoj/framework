"""Keeping a moving average of the weights, and using it once there is one."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, override

from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint

from src.callbacks.moment import at_step

if TYPE_CHECKING:
    from collections.abc import Iterator

    import lightning as L

log = logging.getLogger(__name__)


class EmaWeights(EMAWeightAveraging):
    """An exponential moving average of the weights, validated and saved in their place.

    Lightning's ``EMAWeightAveraging`` does the averaging; what it does not do is
    wait for the average to exist. Its averaged model is a copy of the weights
    taken at ``setup``, and the first update *replaces* rather than blends — so
    until that update runs, the copy holds untrained weights while three of
    Lightning's hooks use it regardless. Each of the four overrides below stands
    down until the average is real, which leaves the live weights in charge:

    - validation would otherwise report the untrained copy,
    - a checkpoint would store it under a metric the live weights earned,
    - and a warmup longer than the run would end by overwriting the trained
      model with it.

    Parameters:
        decay (float): How much of the average survives each update. Nearer 1
            averages over a longer stretch, so 0.9999 is for long runs and 0.99
            for short ones.
        after (float): Share of the run to train before averaging begins, so the
            average does not start from noise. Resolved against the run's total
            steps, which means it survives a change of epoch count.
        **kwargs: Forwarded to ``EMAWeightAveraging`` — ``device`` (``"cpu"``
            keeps the second copy off the GPU), ``use_buffers``,
            ``update_every_n_steps``.
    """

    def __init__(self, decay: float = 0.999, after: float = 0.0, **kwargs: Any) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EmaWeights decay must be in (0, 1), got {decay}.")
        if not 0.0 <= after < 1.0:
            raise ValueError(f"EmaWeights after must be a share of the run in [0, 1), got {after}.")
        super().__init__(decay=decay, **kwargs)
        self._decay = decay  # the parent folds it into avg_fn and keeps nothing to log
        self._after = after

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        # Lightning assigns ``callbacks`` in ``__init__`` rather than declaring it on the class.
        saving = trainer.callbacks  # type: ignore[attr-defined]
        unaware = (
            one
            for one in saving
            if isinstance(one, ModelCheckpoint) and one.save_weights_only and not isinstance(one, EmaModelCheckpoint)
        )
        if next(unaware, None) is not None:
            raise ValueError(
                "EmaWeights cannot be combined with checkpoint(save_weights_only=true): Lightning "
                "runs a callback's save hook only for full checkpoints, so the file would hold the "
                "live weights while the metric it was chosen by came from the averaged ones. "
                "Use ema_checkpoint instead, or set save_weights_only: false."
            )
        self.update_starting_at_step = int(self._after * trainer.estimated_stepping_batches)
        log.info(
            "Averaging weights with decay %s, from %s.",
            self._decay,
            at_step(trainer, self.update_starting_at_step),
        )

    @property
    def _has_averaged(self) -> bool:
        """Whether an update has run, so the averaged model means something.

        Read through the parent's ``state_dict``, which carries exactly this
        number, rather than the attribute behind it: the checkpoint restores it
        there too, so a resumed run knows it already has an average.
        """
        latest: int = self.state_dict()["latest_update_step"]
        return latest > 0

    @contextmanager
    def averaged_weights(self, pl_module: L.LightningModule) -> Iterator[None]:
        """Hold the averaged weights in the model for the duration, then put the live ones back.

        The same swap Lightning does around validation, offered as a scope so
        anything else that has to read the averaged model — saving a
        weights-only checkpoint, exporting — can borrow it without a second copy
        of the weights. Does nothing while there is no average to lend.
        """
        if not self._has_averaged:
            yield
            return
        self._swap_models(pl_module)
        try:
            yield
        finally:
            self._swap_models(pl_module)

    @override
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._has_averaged:
            super().on_validation_epoch_start(trainer, pl_module)

    @override
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._has_averaged:
            super().on_validation_epoch_end(trainer, pl_module)

    @override
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        if self._has_averaged:
            super().on_save_checkpoint(trainer, pl_module, checkpoint)

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._has_averaged:
            super().on_train_end(trainer, pl_module)
        else:
            log.info("Averaging never started, so the trained weights are kept as they are.")


class EmaModelCheckpoint(ModelCheckpoint):
    """A checkpoint whose weights-only files hold the averaged weights.

    Lightning runs a callback's ``on_save_checkpoint`` only when dumping a full
    checkpoint, so on the weights-only path ``EmaWeights`` never gets to
    substitute its weights and the file would keep the live ones — under a
    metric the averaged ones earned.

    Rather than rebuild the save, this lends the model the averaged weights for
    the length of it and lets Lightning dump as it always does. That is sound
    because by the time a checkpoint is written the validation swap has already
    been undone, so the model holds the live weights (verified). Without an
    ``EmaWeights`` beside it, or when saving in full, behaviour is exactly the
    parent's.
    """

    @override
    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # Lightning assigns ``callbacks`` in ``__init__`` rather than declaring it on the class.
        averaging = trainer.callbacks  # type: ignore[attr-defined]
        ema = next((one for one in averaging if isinstance(one, EmaWeights)), None)
        if ema is None or not self.save_weights_only:
            super()._save_checkpoint(trainer, filepath)
            return
        with ema.averaged_weights(trainer.lightning_module):
            super()._save_checkpoint(trainer, filepath)
