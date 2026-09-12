"""Moving one number of a task's objective over the run — a focal gamma, a label smoothing."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, override

import lightning as L
from torch import Tensor, nn

from src.callbacks.moment import Moment
from src.callbacks.registry import callback_registry
from src.losses import Loss
from src.tracking import series
from src.training import FitProfile, TrainingModule

if TYPE_CHECKING:
    from src.training import Learner

log = logging.getLogger(__name__)

RAMPS: Mapping[str, Callable[[float], float]] = {
    "linear": lambda progress: progress,
    "cosine": lambda progress: (1.0 - math.cos(math.pi * progress)) / 2.0,
}
"""Shapes over progress in ``[0, 1]``; cosine leaves both ends gently and hurries through the middle."""

ANNEALED = "schedule"
"""The family an annealed number is reported under: ``schedule/<task>/<number>``.

Stage-less on purpose — what it is worth is a fact of the epoch and not of a split — which is also why
it is not a ``MetricKey``: that grammar starts with a stage, and a progress table rows only what has one.
"""


def scheduled(epoch: int, window: int, start: float, end: float, shape: Callable[[float], float]) -> float:
    """What the number is worth at an epoch: exactly ``start`` at the first, exactly ``end`` from the
    window's last on. A window of one epoch collapses the ramp into a step.

    Pure in the epoch, which is what lets a resumed run pick the ramp up mid-way rather than start it.
    """
    progress = min(epoch / max(window - 1, 1), 1.0)
    return start + (end - start) * shape(progress)


@callback_registry.register("anneal")
class Anneal(L.Callback):
    """Move a number of one task's objective from where it starts to where it should end up.

    The objective never sees Lightning; this is what knows the epoch. The number is found by walking
    the objective's own module tree, because a term is often a wrapper around a library's module and
    the number sits a level below the name it reports under.

    Parameters:
        task: The task whose objective holds the number.
        parameter: The number, optionally prefixed by the term holding it (``focal.gamma``) where more
            than one term of a weighted sum holds one by that name.
        start: What it is worth at the first epoch, whatever it was constructed with.
        end: What it reaches, and holds from there on.
        schedule: The shape of the ramp — ``linear`` or ``cosine``.
        over: How much of the run the ramp spans; the default takes all of it.
    """

    def __init__(
        self,
        task: str,
        parameter: str,
        *,
        start: float,
        end: float,
        schedule: str = "linear",
        over: float = 1.0,
    ) -> None:
        super().__init__()
        if schedule not in RAMPS:
            raise ValueError(f"Anneal knows no {schedule!r} ramp; it shapes one as {', '.join(sorted(RAMPS))}.")
        self._task = task
        self._term, _, self._attribute = parameter.rpartition(".")
        self._start, self._end = float(start), float(end)
        self._schedule = schedule  # kept for the announcement: a shape cannot name itself
        self._over = Moment(over, knob="over")
        self._owner: nn.Module | None = None
        self._window = 0

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        reached = self._over.in_epochs(FitProfile.of(trainer))
        self._window = reached.epoch
        self._owner = self._holder(self._objective(pl_module))
        # A ramp leaves no trace of its own, so a run that ends with a different objective than it
        # started with would say nothing about why. Said once, in the shape every boundary is said in.
        log.info(
            "Annealing %s of task %r from %s to %s, %s, reaching it at %s.",
            self._attribute,
            self._task,
            self._start,
            self._end,
            self._schedule,
            reached,
        )

    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._owner is None:
            return
        value = scheduled(trainer.current_epoch, self._window, self._start, self._end, RAMPS[self._schedule])
        setattr(self._owner, self._attribute, value)
        pl_module.log(series(ANNEALED, series(self._task, self._attribute)), value)

    def _objective(self, pl_module: L.LightningModule) -> Loss:
        """The task's objective, asked of the learner rather than looked for in the module's tree.

        Asked, so the answer follows wherever an algorithm keeps it; a learner that holds none says so,
        and this refuses with what that means rather than with an attribute error three frames down.
        """
        learner: Learner | None = pl_module.learner if isinstance(pl_module, TrainingModule) else None
        objective = learner.loss_of(self._task) if learner is not None else None
        if objective is None:
            raise ValueError(
                f"Anneal moves a number of a task's own objective, and this run holds none for "
                f"{self._task!r}: a model that arrives whole owns its loss internally."
            )
        return objective

    def _holder(self, objective: Loss) -> nn.Module:
        """The one module under the objective holding that number as a plain number.

        One walk whether a term was named or not: every candidate is paired with the name of the term
        it sits under, and naming a term just filters on that — so what a refusal suggests is exactly
        what resolution then accepts.
        """
        modules = dict(objective.named_modules())
        holders: list[tuple[str, nn.Module]] = [
            (_term_of(modules, path), module)
            for path, module in modules.items()
            if _is_a_plain_number(getattr(module, self._attribute, None))
        ]
        if self._term:
            holders = [(term, module) for term, module in holders if term == self._term]
        if len(holders) == 1:
            return holders[0][1]
        if len(holders) > 1:
            named = sorted({term for term, _ in holders if term})
            raise ValueError(
                f"Anneal found {self._attribute!r} in several terms of task {self._task!r}. Say which one "
                "by its reporting name: " + ", ".join(f"'{term}.{self._attribute}'" for term in named)
            )
        raise ValueError(self._nothing_to_move(objective, modules))

    def _nothing_to_move(self, objective: Loss, modules: Mapping[str, nn.Module]) -> str:
        optimized = next(
            (
                type(held).__name__
                for module in modules.values()
                if isinstance(held := getattr(module, self._attribute, None), Tensor)
            ),
            None,
        )
        if optimized is not None:
            return (
                f"Anneal cannot move {self._attribute!r} of task {self._task!r}: it is a {optimized}, and "
                "writing over one fights the optimizer that owns it. Ramps move plain numbers."
            )
        under = f" under term {self._term!r}" if self._term else ""
        return (
            f"Anneal found no number called {self._attribute!r} on the objective of task {self._task!r}"
            f"{under}. {type(objective).__name__} holds: {', '.join(_numbers_of(modules)) or 'none'}."
        )


def _is_a_plain_number(held: object) -> bool:
    """A number a ramp may write over: not a flag, and not something the optimizer owns."""
    return isinstance(held, int | float) and not isinstance(held, bool)


def _numbers_of(modules: Mapping[str, nn.Module]) -> list[str]:
    return sorted(
        {
            name
            for module in modules.values()
            for name, value in vars(module).items()
            if _is_a_plain_number(value) and not name.startswith("_")
        }
    )


def _term_of(modules: Mapping[str, nn.Module], path: str) -> str:
    """The reporting name of the innermost term the module at ``path`` sits under."""
    steps = path.split(".") if path else []
    for depth in range(len(steps), -1, -1):
        enclosing = modules[".".join(steps[:depth])]
        if isinstance(enclosing, Loss):
            return str(enclosing.log_name)
    return ""
