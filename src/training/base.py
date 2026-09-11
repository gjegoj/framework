"""The algorithm boundary: what a batch becomes before anything is optimized."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn

from src.core import Batch, StepOutput
from src.models import Model
from src.tasks import Task


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
        if not tasks:
            raise ValueError("A learner trains at least one task; none was given.")
        self.model = model
        self.tasks = dict(tasks)

    @abstractmethod
    def step(self, batch: Batch) -> StepOutput:
        """One pass turned into a loss, the predictions to score, and the targets to score them against.

        Never performs backward: the trainer owns that. Nor is it told which stage it is in — a learner
        that trains differently in evaluation reads ``self.training``, which every module already carries.
        """
        raise NotImplementedError
