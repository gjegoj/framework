"""A network that answers with what it was handed, so a test about training is not a test about a backbone."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from src.core import FEATURE_AXIS, Axis, ModelOutput, TensorTree
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


class OwnCosineHead(nn.Module):
    """A head a run wrote itself, answering with the angle to one prototype per class.

    Here so that one promise can be held: an objective that reads angles takes a head at the word it
    declares, rather than recognising a class this framework happens to ship.

    The word, and not the member: ``Representation`` is a ``StrEnum`` precisely so that a run writing
    its own head need not import the framework's vocabulary to speak it, which is what a declaration
    compared by identity rather than by value would quietly refuse.
    """

    reads_axes: ClassVar[tuple[str, ...]] = (Axis.CHANNELS,)
    produces: ClassVar[str] = "cosines"

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features: Tensor) -> Tensor:
        return normalize(features, dim=FEATURE_AXIS) @ normalize(self.prototypes, dim=-1).T
