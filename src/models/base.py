"""Model computations and explicitly optional native-loss and generation capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn

from src.core import ModelOutput, ShapeTree, TensorTree


class Model(nn.Module, ABC):
    @property
    def feature_shapes(self) -> Mapping[str, ShapeTree]:
        """Published features available to additional heads; empty for output-only adapters."""
        return {}

    @abstractmethod
    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        """Normalize external tensor/tuple/library outputs at the adapter boundary."""
        raise NotImplementedError


class ModelWithLoss(ABC):
    @abstractmethod
    def forward_with_loss(self, inputs: Mapping[str, TensorTree], targets: Mapping[str, TensorTree]) -> ModelOutput:
        """One pass; predictions may be unavailable while named native losses are present."""
        raise NotImplementedError


class GenerativeModel(ABC):
    @abstractmethod
    def generate(self, inputs: Mapping[str, TensorTree], options: Mapping[str, object]) -> ModelOutput:
        """Optional generation, never invoked implicitly by training."""
        raise NotImplementedError
