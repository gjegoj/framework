"""The plain multitask objective: what every task is judged by, weighed as the run declared and summed."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from functools import reduce
from typing import cast

import torch
from torch import nn

from src.core import Batch, LossOutput, ModelOutput, StepOutput
from src.losses import Loss
from src.models import Model
from src.tasks import Task
from src.training.base import Learner


class StandardLearner(Learner):
    """One network, one loss per task, one number to descend.

    Losses register under ``losses.<task>``, so a loss carrying parameters of its own — an ArcFace
    margin, a learned uncertainty — is optimized and checkpointed with the model rather than beside it.
    """

    def __init__(self, model: Model, tasks: Mapping[str, Task], losses: Mapping[str, Loss]) -> None:
        super().__init__(model, tasks)
        if set(losses) != set(self.tasks):
            raise ValueError(
                f"One loss per task, and one task per loss: the tasks are {', '.join(sorted(self.tasks))}, "
                f"the losses are {', '.join(sorted(losses))}."
            )
        self.losses = nn.ModuleDict({name: losses[name] for name in self.tasks})

    def step(self, batch: Batch) -> StepOutput:
        output = self.model(batch.inputs)
        losses = [self._weighted(name, task, output, batch) for name, task in self.tasks.items()]
        with torch.no_grad():
            # Metrics and displays never go backward, and a prediction built inside the graph would hold
            # every task's activations alive until the optimizer step.
            predictions = {name: task.postprocess(output) for name, task in self.tasks.items()}
        return StepOutput(
            loss=reduce(operator.add, losses),
            predictions=predictions,
            targets={name: task.metric_view(batch) for name, task in self.tasks.items()},
        )

    def _weighted(self, name: str, task: Task, output: ModelOutput, batch: Batch) -> LossOutput:
        """One task's loss under its own name, scaled by the share of the objective the run gave it."""
        computed = cast(LossOutput, self.losses[name](task.raw(output), task.loss_target(batch)))
        return (computed * task.weight).prefixed(name)
