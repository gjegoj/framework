"""Keeping a moving average of the weights, and using it once there is one."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint

from src.callbacks.moment import Moment
from src.callbacks.registry import callback_registry
from src.training import FitProfile

if TYPE_CHECKING:
    import lightning as L

log = logging.getLogger(__name__)


@callback_registry.register("ema")
class EmaWeights(EMAWeightAveraging):
    """An exponential moving average of the weights, validated and saved in their place.

    Lightning's ``EMAWeightAveraging`` does the averaging but does not wait for the average to exist:
    its copy holds the weights as they were before training until the first update, and three of its
    hooks use that copy regardless. Each override here stands down while there is no average, so
    validation, a checkpoint, and a warmup longer than the run itself all see the live weights.

    Parameters:
        decay: How much of the average survives each update — 0.9999 over a long run, 0.99 over a short
            one; the closer to 1, the longer the average remembers.
        after: How much of the run to train before averaging begins, resolved against its total steps.
        **options: Forwarded to ``EMAWeightAveraging``, so every knob of it stays reachable —
            ``device`` (``"cpu"`` keeps the second copy off the accelerator), ``use_buffers``,
            ``update_every_n_steps``.
    """

    def __init__(self, decay: float = 0.999, after: float = 0.0, **options: Any) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(
                f"An EMA decay is the share of the average that survives an update, in (0, 1); got {decay}."
            )
        super().__init__(decay=decay, **options)
        self._decay = decay  # the parent folds it into an averaging function and keeps nothing to say
        self._after = Moment(after, knob="after", opens=True)

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        self._refuse_a_saver_that_would_write_the_live_weights(trainer)
        begins = self._after.in_steps(FitProfile.of(trainer))
        self.update_starting_at_step = begins.step
        log.info("Averaging the weights with decay %s, from %s.", self._decay, begins)

    @property
    def _averaged(self) -> bool:
        """Whether an update has run, which is what makes the averaged model mean anything.

        Read through the parent's ``state_dict``, which carries exactly this number, rather than the
        attribute behind it: a checkpoint restores it there too, so a resumed run knows it already has
        an average and does not stand down a second time.
        """
        latest: int = self.state_dict()["latest_update_step"]
        return latest > 0

    @override
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._averaged:
            super().on_validation_epoch_start(trainer, pl_module)

    @override
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._averaged:
            super().on_validation_epoch_end(trainer, pl_module)

    @override
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        if self._averaged:
            super().on_save_checkpoint(trainer, pl_module, checkpoint)

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._averaged:
            super().on_train_end(trainer, pl_module)
        else:
            log.info("Averaging never began, so the weights the run trained are the ones it keeps.")

    @staticmethod
    def _refuse_a_saver_that_would_write_the_live_weights(trainer: L.Trainer) -> None:
        """A weights-only checkpoint beside an average is a file that says one thing and holds another.

        Measured on lightning 2.6.5: a callback's ``on_save_checkpoint`` runs only for full
        checkpoints, so on the weights-only path the average is never substituted — and the file would
        hold the live weights while the metric it was chosen by came from the averaged ones.
        """
        if any(isinstance(one, ModelCheckpoint) and one.save_weights_only for one in trainer.checkpoint_callbacks):
            raise ValueError(
                "An average of the weights cannot be kept by a checkpoint declaring save_weights_only: "
                "the file would hold the live weights while the metric it was chosen by came from the "
                "averaged ones. Declare save_weights_only: false — a full checkpoint is what "
                "`run.resume_path` continues from anyway."
            )
