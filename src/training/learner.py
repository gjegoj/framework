"""The plain multitask objective: what every task is judged by, weighed as the run declared and summed."""

from __future__ import annotations

import operator
from collections.abc import Collection, Iterable, Mapping
from functools import reduce
from typing import cast, override

import torch
from torch import nn

from src.core import Batch, LossOutput, ModelOutput, StepOutput, as_children
from src.losses import Loss
from src.models import Model
from src.tasks import Task
from src.training.base import Learner
from src.training.registry import learner_registry


@learner_registry.register("standard")
class StandardLearner(Learner):
    """One network, one loss per task, one number to descend.

    Losses register under ``losses.<task>``, so a loss carrying parameters of its own — an ArcFace
    margin, a learned uncertainty — is optimized and checkpointed with the model rather than beside it.
    """

    def __init__(
        self,
        model: Model,
        tasks: Mapping[str, Task],
        losses: Mapping[str, Loss],
        learned_only: Collection[str] = (),
    ) -> None:
        super().__init__(model, tasks)
        if set(losses) != set(self.tasks):
            raise ValueError(
                f"One loss per task, and one task per loss: the tasks are {', '.join(sorted(self.tasks))}, "
                f"the losses are {', '.join(sorted(losses))}."
            )
        if strangers := sorted(set(learned_only) - set(self.tasks)):
            raise ValueError(
                f"{', '.join(strangers)} are not tasks this learner was given, so there is nothing to "
                f"leave unscored; it holds {', '.join(sorted(self.tasks))}."
            )
        self.losses = as_children({name: losses[name] for name in self.tasks})
        self._learned_only = frozenset(learned_only)

    @override
    def loss_of(self, task: str) -> Loss | None:
        """This learner holds one objective per task, under the name the run gave the task.

        Cast, and not ``.get``: torch's ``ModuleDict`` has no such method, and it answers as a module.
        """
        if task not in self.losses:
            return None
        return cast("Loss", self.losses[task])

    def step(self, batch: Batch) -> StepOutput:
        """Every task answers; the ones whose objective only means something while learning are scored then.

        ``self.training`` rather than a stage: a step is not told which one it is in, and the distinction
        this needs is exactly the one every module already carries. A run whose every objective is a
        learning device answers with no loss at all outside training, which the loop is written for —
        and it is what keeps a total from meaning one thing in training and another in evaluation.
        """
        output = self.model(batch.inputs)
        terms = self._terms(output, batch)
        with torch.no_grad():
            # Metrics and displays never go backward, and a prediction built inside the graph would hold
            # every task's activations alive until the optimizer step.
            predictions = {
                name: task.postprocess(output, self.model.produces(name)) for name, task in self.tasks.items()
            }
        return StepOutput(
            loss=reduce(operator.add, terms) if terms else None,
            predictions=predictions,
            targets={name: task.metric_view(batch) for name, task in self.tasks.items()},
        )

    def _terms(self, output: ModelOutput, batch: Batch) -> list[LossOutput]:
        """Everything this step descends, each under the name of the task it is about.

        The one place a learner descending something besides its targets extends — a second network to
        agree with — so that the tasks go on being scored exactly as they are here rather than beside it.
        """
        return [self._weighted(name, task, output, batch) for name, task in self._scored().items()]

    def _scored(self) -> Mapping[str, Task]:
        """The tasks this stage has an objective for, which outside training is not all of them.

        One home, because a learner adding a term of its own has to add it for the same tasks: a total
        made of one set while training and another while validating is two numbers under one name.
        """
        if self.training:
            return self.tasks
        return {name: task for name, task in self.tasks.items() if name not in self._learned_only}

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        """The model's parts for this task, plus its loss — an angular margin keeps the class prototypes."""
        return [*super().parameters_of(task), *self.losses[task].parameters()]

    def _weighted(self, name: str, task: Task, output: ModelOutput, batch: Batch) -> LossOutput:
        """One task's loss under its own name, scaled by the share of the objective the run gave it."""
        computed = cast(LossOutput, self.losses[name](task.raw(output), task.loss_target(batch)))
        return (computed * task.weight).prefixed(name)
