"""Freeze / unfreeze callback — freeze backbone layers for the first N epochs.

Uses Lightning's ``BaseFinetuning`` which correctly handles ``requires_grad``
and re-adds unfrozen params to the optimizer's param groups.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import BaseFinetuning

log = logging.getLogger(__name__)


class FreezeCallback(BaseFinetuning):
    """Freeze named sub-modules before training; optionally unfreeze later.

    Parameters:
        targets (list[str]): Dot-paths into the ``LightningModule``,
            e.g. ``["model.backbone"]``.
        unfreeze_at (int | float): Epoch index to unfreeze (int), or fraction
            of ``max_epochs`` (float in (0, 1]). ``-1`` → never unfreeze.
        train_bn (bool): Keep BatchNorm layers in train mode while frozen
            so their running statistics continue to update.
    """

    def __init__(
        self,
        targets: list[str],
        unfreeze_at: int | float = -1,
        train_bn: bool = False,
    ) -> None:
        super().__init__()
        if not targets:
            raise ValueError("FreezeCallback: targets must be a non-empty list of module paths.")
        if isinstance(unfreeze_at, float) and not (0.0 < unfreeze_at <= 1.0):
            raise ValueError(f"FreezeCallback: unfreeze_at as a fraction must be in (0, 1], got {unfreeze_at}.")
        self._targets = targets
        self._unfreeze_at = unfreeze_at
        self._train_bn = train_bn
        self._unfreeze_epoch: int | None = None

    # ---------------------------------------------------------------- setup

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._unfreeze_epoch = self._resolve_epoch(trainer.max_epochs or 0)
        super().on_fit_start(trainer, pl_module)

    def _resolve_epoch(self, max_epochs: int) -> int | None:
        if self._unfreeze_at == -1 or max_epochs <= 0:
            return None
        if isinstance(self._unfreeze_at, float):
            return int(max_epochs * self._unfreeze_at)
        return int(self._unfreeze_at)

    # -------------------------------------------- BaseFinetuning interface

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        for path in self._targets:
            module = self._resolve(pl_module, path)
            self.freeze(module, train_bn=self._train_bn)
        log.info("Frozen: %s — will unfreeze at epoch %s.", self._targets, self._unfreeze_epoch)

    def finetune_function(self, pl_module: L.LightningModule, current_epoch: int, optimizer: Any) -> None:
        if self._unfreeze_epoch is None or current_epoch < self._unfreeze_epoch:
            return
        for path in self._targets:
            module = self._resolve(pl_module, path)
            self.unfreeze_and_add_param_group(module, optimizer)
        self._unfreeze_epoch = None
        log.info("Unfrozen: %s.", self._targets)

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _resolve(pl_module: L.LightningModule, path: str) -> Any:
        try:
            return pl_module.get_submodule(path)
        except AttributeError as error:
            raise ValueError(f"Cannot resolve module path '{path}' on {type(pl_module).__name__}.") from error
