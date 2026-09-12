"""Holding part of a model still while the rest learns."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from lightning.pytorch.callbacks import BaseFinetuning

from src.callbacks.moment import Boundary, Moment
from src.callbacks.registry import callback_registry
from src.training import FitProfile, module_at

if TYPE_CHECKING:
    import lightning as L
    from torch.optim import Optimizer

log = logging.getLogger(__name__)


@callback_registry.register("freeze")
class Freeze(BaseFinetuning):
    """Hold named parts of the model still, and let them go partway through the run if asked.

    Built on Lightning's ``BaseFinetuning`` for the one thing that is not a flag: freezing has to
    leave normalisation alone where a run asks it to, and that is its ``freeze``. Letting go *is*
    just a flag here, because the parameter groups are the learner's and hold every parameter from
    the first step — named, at the rate its own group declares. Lightning's own
    ``unfreeze_and_add_param_group`` would find them already there, say so, and skip the add.

    Parameters:
        modules: Dot-paths of what to hold still, relative to the model — ``backbone``,
            ``heads.species``. Relative to the *model* rather than to the training module, so that
            nothing a run wraps around it moves a path a config wrote.
        until: How long they are held, as a share of the run or a whole epoch index; the default
            holds them for all of it.
        train_bn: Let normalisation layers keep learning their running statistics while the rest is
            held — those statistics describe *this* data, not the data the weights arrived from.
    """

    def __init__(self, modules: Sequence[str], until: float = 1.0, train_bn: bool = True) -> None:
        super().__init__()
        if not modules:
            raise ValueError("Freeze holds named parts of the model still, and this declaration named none.")
        self._modules = tuple(modules)
        self._until = Moment(until, knob="until")
        self._train_bn = train_bn
        self._release: Boundary | None = None
        self._let_go = False

    @override
    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        for path in self._modules:
            self.freeze(module_at(pl_module, path, reader=type(self).__name__), train_bn=self._train_bn)

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Resolve the hold against the run's length, and say it once.

        Here rather than where the freezing happens: ``BaseFinetuning`` freezes from ``setup``, which
        Lightning calls once per stage, so announcing it there says "frozen until ..." again as the
        test pass begins — about a run that has already finished training.
        """
        super().on_fit_start(trainer, pl_module)
        self._release = self._until.in_epochs(FitProfile.of(trainer))
        self._let_go = False
        log.info("Frozen until %s: %s", self._release, ", ".join(self._modules))

    @override
    def finetune_function(self, pl_module: L.LightningModule, epoch: int, optimizer: Optimizer) -> None:
        """Let go once the run has reached the release, and only once.

        The release is a threshold rather than the one epoch that equals it, because a continued run
        starts *after* the epoch it was tied to: ``BaseFinetuning`` freezes again from every ``setup``,
        so an equality would hold a resumed run for the rest of its life and never say so. The epoch
        logged is the one it happened at, which on such a run is not the one that was declared.

        The optimizer is handed nothing: what was held is already in one of its groups.
        """
        if self._let_go or self._release is None or epoch < self._release.epoch:
            return
        self._let_go = True
        for path in self._modules:
            self.make_trainable(module_at(pl_module, path, reader=type(self).__name__))
        log.info("Unfrozen at epoch %s: %s", epoch, ", ".join(self._modules))
