"""The algorithm boundary: what a batch becomes before anything is optimized, and what optimizing it needs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, NotRequired, Protocol, Self, TypedDict, runtime_checkable

from torch import nn
from torch.optim import Optimizer

from src.core import Batch, StepOutput
from src.losses import Loss
from src.models import Model
from src.tasks import Task

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.utilities.types import LRSchedulerConfigType

SHARED = "backbone"
"""What the group holding everything no task claimed is called.

Defined by subtraction — what is left once every task has taken its own parts — and named for what
that remainder usually is: the encoder. A label for a chart, not an address: the dot-path a config
freezes by is the model's own, and renaming an attribute there must not rename a line in a graph.
"""


class ParameterGroup(TypedDict):
    """One named group as torch takes it; a group without ``lr`` inherits the optimizer's."""

    name: str
    params: list[nn.Parameter]
    lr: NotRequired[float]


class Learner(nn.Module, ABC):
    """How a batch becomes a loss and the predictions to judge it by — the one thing a training loop asks for.

    The learner owns the objective, the model owns the network, the trainer owns backward. That split is
    what lets one model be exported, evaluated or distilled without dragging a loss along, and what lets
    a second algorithm (distillation, an adversarial pair) replace the step without touching either.

    The module paths are part of the contract, not an implementation detail: the network sits under
    ``model``, which is what a checkpoint's keys and a freeze callback's dot-path are written against.
    """

    model: Model
    tasks: Mapping[str, Task]

    def __init__(self, model: Model, tasks: Mapping[str, Task]) -> None:
        super().__init__()
        self.model = model
        self.tasks = dict(tasks)

    def loss_of(self, task: str) -> Loss | None:
        """The objective this learner optimizes one task by, where it holds one of its own.

        Nothing by default, because an algorithm need not: a model that arrives whole owns its loss
        internally, and a callback that schedules a number on one does without rather than failing.
        """
        return None

    @abstractmethod
    def step(self, batch: Batch) -> StepOutput:
        """One pass turned into a loss, the predictions to score, and the targets to score them against.

        Never performs backward: the trainer owns that. Nor is it told which stage it is in — a learner
        that trains differently in evaluation reads ``self.training``, which every module already carries.
        """
        raise NotImplementedError

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        """Everything that belongs to one task alone; a learner adds whatever it holds beside the model."""
        return self.model.parameters_of(task)

    def parameter_groups(self) -> list[ParameterGroup]:
        """One group per task that owns parameters, and one for everything they share.

        Split whether or not a rate was declared, because the groups are also what a learning-rate
        monitor draws one line each of. A group carries an ``lr`` only where its task declared one:
        the base rate has a single home, the optimizer section.
        """
        groups: list[ParameterGroup] = []
        claimed: set[int] = set()
        for name, task in self.tasks.items():
            owned = list(self.parameters_of(name))
            if not owned:
                self._refuse_a_rate_over_nothing(name, task)
                continue
            claimed.update(id(parameter) for parameter in owned)
            group: ParameterGroup = {"name": name, "params": owned}
            if task.lr is not None:
                group["lr"] = task.lr
            groups.append(group)
        shared = [parameter for parameter in self.parameters() if id(parameter) not in claimed]
        if shared:
            groups.insert(0, {"name": SHARED, "params": shared})
        return groups

    @staticmethod
    def _refuse_a_rate_over_nothing(name: str, task: Task) -> None:
        if task.lr is not None:
            raise ValueError(
                f"Task {name!r} declares lr={task.lr}, but nothing in this run belongs to it alone — "
                "the rate would move nothing. Drop the task's lr, or give it a head of its own."
            )


@dataclass(frozen=True, slots=True)
class FitProfile:
    """What a run turns out to be, once the fit loop exists: how many optimizer steps, over how many epochs.

    Two numbers rather than three: the per-epoch count follows from them, so an incoherent triple stays
    unrepresentable. Only the trainer can answer this, which is why a schedule is built as a factory.
    """

    total_steps: int
    epochs: int

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.epochs < 1:
            raise ValueError(f"A fit runs at least one step and one epoch; got {self.total_steps}/{self.epochs}.")

    @classmethod
    def of(cls, trainer: L.Trainer) -> Self:
        """How long this fit turned out to be — the one question only the trainer can answer.

        Read from ``estimated_stepping_batches`` because that is what Lightning guarantees this early:
        the loops are not set up yet, so the per-epoch counts are still unknown. Everything measured
        against the length of a run comes through here — a schedule, a freeze that lets go partway, an
        average that starts late — so a fit with no declared end is refused in one place.
        """
        steps, epochs = trainer.estimated_stepping_batches, trainer.max_epochs
        # Measured on lightning 2.6.5: a loop with no length reports -1 rather than infinity.
        if not isfinite(steps) or steps < 1 or epochs is None or epochs < 1:
            raise ValueError(
                f"This fit has no declared end — {steps} optimizer steps over {epochs} epochs — and anything "
                "timed against the length of the run needs one. Declare epochs, or drop what needs them."
            )
        return cls(total_steps=int(steps), epochs=epochs)

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps in one epoch; at least one, however short the loop."""
        return max(self.total_steps // self.epochs, 1)


type OptimizerFactory = Callable[[Sequence[ParameterGroup]], Optimizer]
"""Builds an optimizer over named groups — a factory, because the parameters do not exist while config is read.

``partial(torch.optim.AdamW, lr=1e-3)`` satisfies it: every torch constructor takes group dicts, and a
group naming no rate of its own inherits the factory's.
"""

type SchedulerFactory = Callable[[Optimizer, FitProfile], LRSchedulerConfigType]
"""Builds a schedule and the policy Lightning steps it by, once the optimizer and the fit's length are known.

Lightning's own policy type rather than a mapping of ours: it is handed over verbatim, and a key
misspelled on the way there is then a type error rather than a schedule that silently never steps."""


@runtime_checkable
class DeclaresMetricDirections(Protocol):
    """Something that can say which way each of its measurements is better.

    A capability rather than a contract: a display asks for it and does without where it is absent.
    Keyed by series (``task/label``), because a direction belongs to the measurement and holds in
    every stage it is measured in; ``None`` is a reading with no better direction at all.
    """

    def metric_directions(self) -> Mapping[str, bool | None]: ...


@runtime_checkable
class AcceptsBatchTransform(Protocol):
    """Something that will let a declared transform rewrite its training batches.

    A capability rather than a class, because the seam has to belong to whoever owns the batch: a
    ``Batch`` is frozen and Lightning discards whatever a callback's hook returns, so a callback can
    only hand the rewriting over. ``None`` takes it back again.
    """

    def transform_batches(self, transform: Callable[[Batch], Batch] | None) -> None: ...
