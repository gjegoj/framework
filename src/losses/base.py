"""Internal objective context; ordinary user losses keep forward(outputs, targets)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor, nn

from src.core import LossOutput, TensorTree


@dataclass(frozen=True, slots=True)
class LossInput:
    outputs: TensorTree = None
    targets: TensorTree = None
    model_losses: Mapping[str, Tensor] = field(default_factory=dict)


class Loss(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: LossInput) -> LossOutput:
        raise NotImplementedError
