"""Running a batch transform over training batches, for as long as the run wants it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

import lightning as L

from src.callbacks.moment import Boundary, Moment
from src.callbacks.registry import callback_registry
from src.training import FitProfile, TrainingModule
from src.transforms import BatchTransform

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core import Batch

log = logging.getLogger(__name__)


@callback_registry.register("batch_transform")
class ApplyBatchTransform(L.Callback):
    """Rewrite training batches with a declared transform, until a point in the run.

    Installed into the module rather than applied here: a ``Batch`` is frozen and Lightning discards
    whatever a callback's hook returns, so a callback can never replace a batch — it can only say what
    should. What it *does* own is everything else: which transform, bound to which tasks, for how long.

    Bound to the run's tasks at setup, because mixing rewrites every one of their targets — so a task
    the transform cannot serve is refused there, before the first batch rather than an hour into one.

    Parameters:
        transform: What to run, declared by import path — ``{_target_: src.transforms.MixUp,
            alpha: 0.4}`` — the same grammar the pixel pipeline is written in.
        until: How long to keep running it. The default runs it for the whole fit; stopping earlier
            lets a run finish on the data it will be judged on.
    """

    def __init__(self, transform: BatchTransform, until: float = 1.0) -> None:
        super().__init__()
        if not isinstance(transform, BatchTransform):
            raise TypeError(
                f"{type(transform).__name__} is not a batch transform: one is bound to the run's tasks "
                "with for_tasks(tasks) before it rewrites anything, and this answers no such question."
            )
        self._declared = transform
        self._until = Moment(until, knob="until")
        self._bound: Callable[[Batch], Batch] | None = None
        self._stop: Boundary | None = None
        self._stopped = False

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        # Lightning's connector fills `callbacks` rather than the Trainer annotating it, so it is
        # asked for the way `checkpoint_callback` is elsewhere.
        installed: list[L.Callback] = getattr(trainer, "callbacks", [])
        sharing = [one for one in installed if isinstance(one, ApplyBatchTransform)]
        if len(sharing) > 1:
            named = ", ".join(type(one._declared).__name__ for one in sharing)
            raise ValueError(
                f"A run rewrites its batches with one transform, and {len(sharing)} are declared: {named}. "
                "The module keeps a single seam, so these would not compose but overwrite each other, and "
                "the first to reach its 'until' would clear the survivor. Declare one, composing inside it."
            )
        if not isinstance(pl_module, TrainingModule):
            raise TypeError(
                f"{type(self._declared).__name__} rewrites the targets of a run's tasks, and "
                f"{type(pl_module).__name__} holds none to rewrite."
            )
        self._bound = self._declared.for_tasks(list(pl_module.learner.tasks.values()))

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Install it, and say when it will stop before it starts rather than once it has."""
        self._stop = self._until.in_epochs(FitProfile.of(trainer))
        self._stopped = False
        self._install(pl_module, self._bound)
        log.info("%s applied until %s.", type(self._declared).__name__, self._stop)

    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._stopped or self._stop is None or trainer.current_epoch < self._stop.epoch:
            return
        self._stopped = True
        self._install(pl_module, None)
        log.info("%s stopped at epoch %s.", type(self._declared).__name__, trainer.current_epoch)

    @staticmethod
    def _install(pl_module: L.LightningModule, transform: Callable[[Batch], Batch] | None) -> None:
        """Hand the rewriting to whoever owns the batch.

        No second refusal here: ``setup`` already established that this is the module that owns one,
        and a check that cannot fail reads as a guard while guarding nothing.
        """
        assert isinstance(pl_module, TrainingModule)
        pl_module.transform_batches(transform)
