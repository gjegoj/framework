"""The feature-extraction boundary; heads consume its named values without hidden reshaping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn

from src.core import ShapeTree, TensorTree


class Backbone(nn.Module, ABC):
    @property
    @abstractmethod
    def feature_shapes(self) -> Mapping[str, ShapeTree]:
        """Shapes of selected features; adapter-specific extraction settings are resolved beforehand."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, TensorTree]:
        """Return the declared features in one differentiable pass."""
        raise NotImplementedError

    def native_head(self, inputs: tuple[str, ...], output_shape: ShapeTree) -> nn.Module | None:
        """Offer this family's head for selected features, or None when no native head is available.

        The library adapter owns head construction and any carried checkpoint weights.
        Assembly chooses this method only for a native-head declaration; an explicit
        request receiving None fails during assembly rather than silently using another head.
        """
        return None
