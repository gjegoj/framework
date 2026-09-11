"""A network that answers with what it was handed, so a test about training is not a test about a backbone."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from src.core import ModelOutput, TensorTree
from src.models import Model


class Echo(Model):
    """Scales what it was given: one shared parameter and one per task, as a composite family has.

    The shared scale keeps a step's graph visible, and the split between shared and per-task parameters
    is what a learner groups by.
    """

    def __init__(self, outputs: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.ones(()))
        self.heads = nn.ParameterDict({name: nn.Parameter(torch.ones(())) for name in outputs})
        self.answers = dict(outputs)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(
            outputs={name: value * self.shared * self.heads[name] for name, value in self.answers.items()}
        )

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        return [self.heads[task]] if task in self.heads else ()
