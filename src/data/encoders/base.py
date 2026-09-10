"""Raw targets before transforms, encoded targets after transforms; no vocabulary discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar, Self

from src.core import Geometry, TargetInfo, TensorTree


class TargetEncoder(ABC):
    """Load before spatial transforms, encode after them; fit only on training values.

    Class indices are declared, never discovered. Fitted encoders may implement
    data.preprocessing.Stateful for artifact restoration; that is not an encoding operation.
    """

    geometry: ClassVar[Geometry] = Geometry.NONE

    @property
    @abstractmethod
    def info(self) -> TargetInfo:
        """Resolved target facts; specialized encoders may return a typed TargetInfo subclass."""
        raise NotImplementedError

    def load(self, value: object) -> object:
        """Prepare a raw target for sample transforms; file-backed encoders may load a mask."""
        return value

    @abstractmethod
    def encode(self, value: object) -> TensorTree:
        """Convert the transformed target into its training representation."""
        raise NotImplementedError

    def fit(self, values: Iterable[object]) -> Self:
        """No-op for declared/stateless encoders; fitted encoders learn non-vocabulary state on train."""
        return self

    def validate(self, values: Iterable[object]) -> None:
        """Check non-training values without refitting; streaming implementations may validate per sample."""
