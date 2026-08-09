"""Holding part of a model still while the rest learns."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, override

from lightning.pytorch.callbacks import BaseFinetuning

from src.callbacks.moment import at_epoch

if TYPE_CHECKING:
    import lightning as L
    from torch import nn
    from torch.optim import Optimizer

log = logging.getLogger(__name__)


class Freeze(BaseFinetuning):
    """Freeze named sub-modules, optionally letting them go partway through training.

    Built on Lightning's ``BaseFinetuning`` rather than on ``requires_grad``
    directly: unfreezing has to return the parameters to the optimizer's groups,
    and that is the step hand-rolled freezing usually misses — the weights thaw
    but never move.

    Parameters:
        modules (list[str]): Dot-paths to the modules to hold still, relative to
            the training module — ``model.backbone``.
        until (float): How long they are held. A value of 1 or less is a share
            of the run, so the same setting survives a change of epoch count;
            a whole number above 1 is an epoch index. The boundary rounds up to
            a whole epoch, so a declared freeze always holds at least one. The
            default holds for the whole run.
        train_bn (bool): Keep normalisation layers learning their running
            statistics while the rest is frozen — usually right, since those
            statistics describe *this* dataset, not the one pretraining used.
    """

    def __init__(self, modules: list[str], until: float = 1.0, train_bn: bool = True) -> None:
        super().__init__()
        if not modules:
            raise ValueError("Freeze needs at least one module to hold still.")
        if until <= 0 or (until > 1 and until != int(until)):
            raise ValueError(f"Freeze until is a share of the run in (0, 1] or a whole epoch index, got {until}.")
        self._modules = list(modules)
        self._until = until
        self._train_bn = train_bn

    def release_epoch(self, max_epochs: int) -> int:
        """The epoch the modules are let go at; ``max_epochs`` means the run ends first."""
        if self._until <= 1:
            return math.ceil(self._until * max_epochs)
        return int(self._until)

    @override
    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        for path in self._modules:
            self.freeze(self._resolve(pl_module, path), train_bn=self._train_bn)

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Announce the hold here rather than where it happens, and only for a fit.

        ``BaseFinetuning`` freezes from ``setup``, which Lightning calls once per
        stage — so announcing it there printed "frozen until ..." again as the test
        pass began, about a run that had already finished training.
        """
        super().on_fit_start(trainer, pl_module)
        release = self.release_epoch(int(trainer.max_epochs or 0))
        log.info("Frozen until %s: %s", at_epoch(trainer, release), ", ".join(self._modules))

    @override
    def finetune_function(self, pl_module: L.LightningModule, epoch: int, optimizer: Optimizer) -> None:
        if epoch != self.release_epoch(int(pl_module.trainer.max_epochs or 0)):
            return
        for path in self._modules:
            self.unfreeze_and_add_param_group(self._resolve(pl_module, path), optimizer)
        log.info("Unfrozen at %s: %s", at_epoch(pl_module.trainer, epoch), ", ".join(self._modules))

    @staticmethod
    def _resolve(root: Any, path: str) -> nn.Module:
        """The sub-module a dot-path names, or a message saying what was there instead."""
        current = root
        for step in path.split("."):
            try:
                current = getattr(current, step)
            except AttributeError:
                available = ", ".join(name for name, _ in current.named_children()) or "none"
                raise LookupError(
                    f"Freeze cannot find '{path}': '{step}' is not a module of "
                    f"{type(current).__name__}. Available: {available}."
                ) from None
        found: nn.Module = current
        return found
