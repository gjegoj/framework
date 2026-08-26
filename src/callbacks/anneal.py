"""Moving one number of a criterion over the run — focal gamma, label smoothing."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, override

import lightning as L
from torch import Tensor, nn

from src.callbacks.moment import at_epoch
from src.core.ports import Model

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

SCHEDULES: dict[str, Callable[[float], float]] = {
    "linear": lambda progress: progress,
    "cosine": lambda progress: (1.0 - math.cos(math.pi * progress)) / 2.0,
}
"""Easings over progress in [0, 1]; cosine leaves both ends gently."""


def scheduled_value(epoch: int, window: int, start: float, end: float, shape: Callable[[float], float]) -> float:
    """The value for an epoch: exactly ``start`` at 0, exactly ``end`` from the window on.

    A window of one collapses the ramp to a step. Pure in ``epoch``, which is
    what makes a resumed run pick the schedule up mid-ramp.
    """
    progress = min(epoch / max(window - 1, 1), 1.0)
    return start + (end - start) * shape(progress)


class AnnealCriterion(L.Callback):
    """Anneal a numeric attribute of one task's criterion over the run.

    The criterion never sees Lightning; this is what knows the epoch, and what it writes is
    a pure function of ``current_epoch``, so a resumed run loses nothing. The attribute is
    found by walking the criterion's module tree; ambiguity is resolved with the part's
    logging name — ``parameter: ce.label_smoothing``.

    Parameters:
        task (str): The task whose criterion is annealed.
        parameter (str): The numeric attribute, optionally prefixed by a part name.
        start (float): Value at epoch 0, overriding the constructed one.
        end (float): Value from the end of the window on.
        schedule (str): Easing — ``linear`` or ``cosine``.
        over (float): Share of the run the ramp spans, in (0, 1].
    """

    def __init__(
        self,
        task: str,
        parameter: str,
        start: float,
        end: float,
        schedule: str = "linear",
        over: float = 1.0,
    ) -> None:
        super().__init__()
        if schedule not in SCHEDULES:
            raise ValueError(f"AnnealCriterion knows no '{schedule}' schedule; available: {sorted(SCHEDULES)}.")
        if not 0.0 < over <= 1.0:
            raise ValueError(f"AnnealCriterion over must be a share of the run in (0, 1], got {over}.")
        self._task = task
        self._part, _, self._attribute = parameter.rpartition(".")
        self._start = float(start)
        self._end = float(end)
        self._shape = SCHEDULES[schedule]
        self._schedule = schedule  # kept for the announcement; the shape cannot name itself
        self._over = over
        self._owner: nn.Module | None = None
        self._window = 0

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        max_epochs = trainer.max_epochs or 0
        if max_epochs <= 0:
            raise ValueError(
                "AnnealCriterion counts in epochs, and this trainer declares no max_epochs. "
                "Set trainer.max_epochs, or drop the schedule."
            )
        self._window = max(1, round(self._over * max_epochs))
        self._owner = self._find_owner(self._criterion_of(pl_module))
        # The ramp is a pure function of the epoch and leaves no trace of its own,
        # so a run that ends with a different loss than it started with would say
        # nothing about why. Announced once, in the shape every other callback uses
        # for a boundary.
        log.info(
            "Annealing %s of task '%s' from %s to %s, %s, reaching it at %s.",
            self._attribute,
            self._task,
            self._start,
            self._end,
            self._schedule,
            at_epoch(trainer, self._window - 1),
        )

    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._owner is None:
            return
        value = scheduled_value(trainer.current_epoch, self._window, self._start, self._end, self._shape)
        setattr(self._owner, self._attribute, value)
        pl_module.log(f"schedule/{self._task}/{self._attribute}", value)

    def _criterion_of(self, pl_module: L.LightningModule) -> nn.Module:
        """The task's criterion, asked of the model rather than looked for in its tree.

        Through the port, so the answer follows the model wherever this run nested it:
        reading ``model.criteria`` reached the composite family only, and a distilled
        run — which wraps the student one level down — died here before its first batch
        although both sections are supported. An unknown task is the model's own
        refusal, naming the tasks it does compose.
        """
        model = getattr(pl_module, "model", None)
        criterion = model.criterion_of(self._task) if isinstance(model, Model) else None
        if criterion is None:
            raise ValueError(
                f"AnnealCriterion schedules a number on a task's own criterion, and "
                f"{type(model).__name__} composes none — a vendor family owns its loss internally."
            )
        return criterion

    def _find_owner(self, criterion: nn.Module) -> nn.Module:
        """The one module holding the attribute as a plain number, found by torch's walk.

        One uniform scan whether a part prefix was given or not: every owner is
        paired with the logging name of the part it sits under, and the prefix
        just filters on that name — so what the error message suggests is
        exactly what resolution accepts.
        """
        modules = dict(criterion.named_modules())
        owners: list[tuple[str, nn.Module]] = []
        refused: str | None = None
        for name, module in modules.items():
            held = getattr(module, self._attribute, None)
            if isinstance(held, bool) or held is None:
                continue
            if isinstance(held, (int, float)):
                owners.append((_enclosing_part(modules, name), module))
            elif isinstance(held, (nn.Parameter, Tensor)):
                refused = type(held).__name__
        if self._part:
            owners = [(part, module) for part, module in owners if part == self._part]
        if len(owners) == 1:
            return owners[0][1]
        if len(owners) > 1:
            parts = sorted({part for part, _ in owners if part})
            raise ValueError(
                f"AnnealCriterion found '{self._attribute}' in several parts of task '{self._task}'. "
                f"Say which one with its logging name: " + ", ".join(f"'{part}.{self._attribute}'" for part in parts)
            )
        if refused is not None:
            raise ValueError(
                f"AnnealCriterion cannot schedule '{self._attribute}' of task '{self._task}': it is a "
                f"{refused}, and writing over one fights the optimizer. Schedule plain numbers only."
            )
        available = sorted(
            {
                name
                for module in modules.values()
                for name, value in vars(module).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not name.startswith("_")
            }
        )
        raise ValueError(
            f"AnnealCriterion found no numeric '{self._attribute}' on the criterion of task "
            f"'{self._task}'"
            + (f" under part '{self._part}'" if self._part else "")
            + f". Numeric attributes: {', '.join(available) or 'none'}."
        )


def _enclosing_part(modules: dict[str, nn.Module], name: str) -> str:
    """The logging name of the innermost part the module ``name`` sits under."""
    steps = name.split(".") if name else []
    for depth in range(len(steps), -1, -1):
        part = getattr(modules[".".join(steps[:depth])], "part_name", None)
        if isinstance(part, str):
            return part
    return ""
