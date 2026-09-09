"""Holding part of a model still while the rest learns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from lightning.pytorch.callbacks import BaseFinetuning

from src.callbacks.moment import at_epoch, declared_moment, epoch_at
from src.models import without_teachers
from src.training.module import TrainingModule

if TYPE_CHECKING:
    import lightning as L
    from torch import nn
    from torch.optim import Optimizer

log = logging.getLogger(__name__)


class Freeze(BaseFinetuning):
    """Freeze named sub-modules, optionally letting them go partway through training.

    Built on Lightning's ``BaseFinetuning``: unfreezing has to return the parameters to the
    optimizer's groups, the step hand-rolled freezing misses.

    Parameters:
        modules (list[str]): Dot-paths to the modules to hold still, relative to the model
            that ships — ``backbone``, ``heads.tags.base``. Relative to the *model* rather
            than to the training module, so no scaffolding around it (a distilled run's
            teachers) moves a path: in a distilled run ``backbone`` is the student's.
        until (float): How long they are held. A value of 1 or less is a share of the run;
            a whole number above 1 is an epoch index. Rounds up to a whole epoch; the
            default holds for the whole run.
        train_bn (bool): Keep normalisation layers learning their running statistics while
            the rest is frozen — those statistics describe *this* dataset.
    """

    def __init__(self, modules: list[str], until: float = 1.0, train_bn: bool = True) -> None:
        super().__init__()
        if not modules:
            raise ValueError("Freeze needs at least one module to hold still.")
        self._modules = list(modules)
        self._until = declared_moment(until, owner="Freeze", knob="until", role="end")
        self._train_bn = train_bn

    def release_epoch(self, max_epochs: int) -> int:
        """The epoch the modules are let go at; ``max_epochs`` means the run ends first."""
        return epoch_at(self._until, max_epochs)

    @override
    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        for path in self._modules:
            self.freeze(self._resolve(_shipped(pl_module), path), train_bn=self._train_bn)

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
            self.unfreeze_and_add_param_group(self._resolve(_shipped(pl_module), path), optimizer)
        log.info("Unfrozen at %s: %s", at_epoch(pl_module.trainer, epoch), ", ".join(self._modules))

    @staticmethod
    def _resolve(root: Any, path: str) -> nn.Module:
        """The sub-module a dot-path names, or a message saying what was there instead.

        A path written for the old spelling — from the training module, ``model.backbone`` —
        is recognised where it would otherwise fail, and answered with the new one.
        """
        current = root
        for position, step in enumerate(path.split(".")):
            try:
                current = getattr(current, step)
            except AttributeError:
                available = ", ".join(name for name, _ in current.named_children()) or "none"
                if position == 0 and step == TrainingModule.MODEL:
                    relative = path.removeprefix(f"{TrainingModule.MODEL}.")
                    raise LookupError(
                        f"Freeze paths are relative to the model that ships, not to the training module: "
                        f"write '{relative}', not '{path}'."
                    ) from None
                raise LookupError(
                    f"Freeze cannot find '{path}': '{step}' is not a module of "
                    f"{type(current).__name__}. Available: {available}."
                ) from None
        found: nn.Module = current
        return found


def _shipped(pl_module: L.LightningModule) -> nn.Module:
    """The model the paths are relative to: the module's model, minus a distilled run's scaffolding."""
    model: Any = pl_module.model  # nn.Module types every attribute as Tensor | Module
    shipped: nn.Module = without_teachers(model)
    return shipped
