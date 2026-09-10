"""The algorithm boundary; Lightning alone owns backward and optimizer stepping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import NotRequired, TypedDict

from torch import nn

from src.core import Batch, Stage, StepOutput
from src.models import Model
from src.tasks import Task


class ParameterGroup(TypedDict):
    name: str
    params: list[nn.Parameter]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]


class TrainingStrategy(nn.Module, ABC):
    @property
    @abstractmethod
    def model(self) -> Model: ...
    @property
    @abstractmethod
    def tasks(self) -> Mapping[str, Task]: ...

    @abstractmethod
    def step(self, batch: Batch, stage: Stage) -> StepOutput:
        """Train requires loss; metric-only evaluation is valid. Never perform backward here."""
        raise NotImplementedError

    @abstractmethod
    def parameter_groups(self) -> list[ParameterGroup]:
        """Every trainable parameter exactly once; conflicting assignments fail by identity."""
        raise NotImplementedError
