"""Epoch-wise schedule for one numeric criterion attribute (e.g. FocalLoss ``gamma``).

The criterion stays a dumb brick (Dependency Rule: losses never see Lightning);
this callback — like ``FreezeCallback`` / ``BatchTransformCallback`` / ``EmaCallback``
— is the humble object that knows the epoch. ``start`` overrides the criterion's
constructed value from epoch 0: the schedule is the single source of truth for the
parameter during fit. The scheduled value is a plain float attribute — invisible to
EMA weight swaps and checkpoints — and a pure function of ``current_epoch``, so the
callback is stateless and resume-safe.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, cast

import lightning as L
import torch

log = logging.getLogger(__name__)

_SCHEDULES: dict[str, Callable[[float], float]] = {
    "linear": lambda progress: progress,
    "cosine": lambda progress: (1.0 - math.cos(math.pi * progress)) / 2.0,
}


def scheduled_value(epoch: int, window: int, start: float, end: float, shape: Callable[[float], float]) -> float:
    """Interpolate the scheduled value for an epoch.

    Epoch 0 yields exactly ``start``; the last epoch of the window (and every epoch
    after) yields exactly ``end``. Degenerate ``window == 1`` has no room to ramp —
    the value stays at ``start``.

    Parameters:
        epoch (int): Current epoch index (0-based).
        window (int): Number of epochs the ramp spans (``>= 1``).
        start (float): Value at epoch 0.
        end (float): Value at the end of the window.
        shape (Callable[[float], float]): Easing over progress in ``[0, 1]``.
    """
    progress = min(epoch / max(window - 1, 1), 1.0)
    return start + (end - start) * shape(progress)


class CriterionScheduleCallback(L.Callback):
    """Anneal a numeric attribute of one task's criterion over epochs.

    Parameters:
        task (str): Name of the task whose criterion is scheduled.
        parameter (str): Numeric attribute to anneal — resolved on the criterion
            itself, then on its wrapped ``_loss`` (the ``SingleTermCriterion`` case).
        start (float): Value applied at epoch 0 (overrides the constructed value).
        end (float): Value reached at the end of the schedule window.
        schedule (str): Easing kind — ``"linear"`` or ``"cosine"``.
        over (float): Fraction of ``max_epochs`` the ramp spans, in ``(0, 1]``.
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
        if schedule not in _SCHEDULES:
            raise ValueError(f"Unknown schedule kind {schedule!r}; available: {sorted(_SCHEDULES)}.")
        if not 0.0 < over <= 1.0:
            raise ValueError(f"over must be a fraction in (0, 1], got {over}.")
        for label, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number, got {value!r}.")
        self._task_name = task
        self._parameter = parameter
        self._leaf_attribute = parameter.split(".")[-1]
        self._start = float(start)
        self._end = float(end)
        self._shape = _SCHEDULES[schedule]
        self._over = over
        self._owner: object | None = None
        self._window: int | None = None

    # ---------------------------------------------------------------- setup

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        criterion = self._find_criterion(pl_module)
        self._owner = self._resolve_owner(criterion)
        total_epochs = trainer.max_epochs
        if total_epochs is None or total_epochs <= 0:
            log.warning(
                "CriterionScheduleCallback: max_epochs is not set — schedule for '%s.%s' is disabled.",
                self._task_name,
                self._parameter,
            )
            self._window = None
            return
        self._window = max(1, round(self._over * total_epochs))

    def _find_criterion(self, pl_module: L.LightningModule) -> object:
        tasks = getattr(pl_module, "tasks", None)
        if tasks is None:
            raise ValueError(f"CriterionScheduleCallback: {type(pl_module).__name__} exposes no 'tasks'.")
        for task in tasks:
            if task.name == self._task_name:
                return task.criterion
        task_names = [task.name for task in tasks]
        raise ValueError(
            f"CriterionScheduleCallback: unknown task {self._task_name!r}; configured tasks: {task_names}."
        )

    def _resolve_owner(self, criterion: object) -> object:
        *term_path, leaf = self._parameter.split(".")
        for segment in term_path:
            criterion = self._descend(criterion, segment)
        candidates: list[object] = [criterion]
        wrapped = getattr(criterion, "_loss", None)
        if wrapped is not None:
            candidates.append(wrapped)
        for owner in candidates:
            if not hasattr(owner, leaf):
                continue
            value = getattr(owner, leaf)
            if isinstance(value, torch.nn.Parameter):
                raise ValueError(
                    f"CriterionScheduleCallback: {self._parameter!r} is a learnable nn.Parameter — "
                    "scheduling it fights the optimizer."
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"CriterionScheduleCallback: {self._parameter!r} is not a plain numeric attribute "
                    f"(got {type(value).__name__}); tensors/buffers are not schedulable."
                )
            return owner
        available = sorted(
            {
                name
                for owner in candidates
                for name, value in vars(owner).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        message = (
            f"CriterionScheduleCallback: criterion for task {self._task_name!r} has no attribute "
            f"{leaf!r}. Numeric attributes available: {available}."
        )
        term_keys = getattr(criterion, "keys", None)
        if not term_path and callable(term_keys):
            examples = ", ".join(f"'{term}.{leaf}'" for term in sorted(term_keys()))
            message += f" This is a composite criterion — address a term's parameter with a dot-path, e.g. {examples}."
        raise ValueError(message)

    @staticmethod
    def _descend(criterion: object, segment: str) -> object:
        """Step one dot-path segment down into a composite criterion's term."""
        try:
            return cast("Any", criterion)[segment]
        except KeyError as error:
            raise ValueError(f"CriterionScheduleCallback: {error.args[0]}") from error
        except TypeError as error:
            raise ValueError(
                f"CriterionScheduleCallback: cannot descend into {segment!r} — "
                f"{type(criterion).__name__} is not a composite criterion."
            ) from error

    # ----------------------------------------------------------- application

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._owner is None or self._window is None:
            return
        value = scheduled_value(trainer.current_epoch, self._window, self._start, self._end, self._shape)
        setattr(self._owner, self._leaf_attribute, value)
        pl_module.log(f"schedule/{self._task_name}/{self._parameter}", value)
        log.debug("Scheduled %s.%s = %.4f (epoch %d).", self._task_name, self._parameter, value, trainer.current_epoch)
