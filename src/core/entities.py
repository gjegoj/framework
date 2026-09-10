"""Values exchanged by data, models, tasks and reporting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import cast

import torch
from torch import Tensor

from src.core.types import ShapeTree, TensorTree


@dataclass(frozen=True, slots=True)
class Sample:
    inputs: Mapping[str, object]
    targets: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    auxiliary_inputs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Batch:
    """The collator supplies example count; ragged tensor lengths do not determine it."""

    inputs: Mapping[str, TensorTree]
    count: int = field(kw_only=True)
    targets: Mapping[str, TensorTree] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Batch count must be a positive integer.")

    def __len__(self) -> int:
        return self.count

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Batch:
        return self._map(lambda tensor: tensor.to(device=device, non_blocking=non_blocking))

    def detach(self) -> Batch:
        return self._map(Tensor.detach)

    def _map(self, operation: Callable[[Tensor], Tensor]) -> Batch:
        from lightning_utilities.core.apply_func import apply_to_collection

        # Frozen dataclass traversal in lightning_utilities 0.15.3 loses field replacements.
        return Batch(
            inputs=cast(Mapping[str, TensorTree], apply_to_collection(self.inputs, Tensor, operation)),
            targets=cast(Mapping[str, TensorTree], apply_to_collection(self.targets, Tensor, operation)),
            count=len(self),
            metadata=self.metadata,
        )


def validate_classes(classes: Mapping[int, str]) -> None:
    """Validate the user-defined output vocabulary, never infer it from observations."""
    if not classes or any(type(index) is not int for index in classes):
        raise ValueError("Classes require integer indices.")
    if set(classes) != set(range(len(classes))):
        raise ValueError("Class indices must be contiguous from zero.")
    if any(not isinstance(name, str) or not name.strip() for name in classes.values()):
        raise ValueError("Class names must be nonblank strings.")


@dataclass(frozen=True, slots=True)
class InputInfo:
    shape: ShapeTree
    modality: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TargetInfo:
    shape: ShapeTree = None
    classes: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        if self.classes is not None:
            validate_classes(self.classes)

    @property
    def num_classes(self) -> int | None:
        return None if self.classes is None else len(self.classes)


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    inputs: Mapping[str, InputInfo]
    targets: Mapping[str, TargetInfo]
    splits: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


type Prediction = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Raw losses never change under weighting; contributions describe the actual objective."""

    total: Tensor
    losses: Mapping[str, Tensor] = field(default_factory=dict)
    contributions: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total.ndim != 0:
            raise ValueError("Loss total must be a scalar tensor.")

    def __add__(self, other: LossOutput) -> LossOutput:
        duplicates = (self.losses.keys() & other.losses.keys()) | (
            self.contributions.keys() & other.contributions.keys()
        )
        if duplicates:
            raise ValueError(f"Loss names need distinct names: {sorted(duplicates)}.")
        return LossOutput(
            self.total + other.total, {**self.losses, **other.losses}, {**self.contributions, **other.contributions}
        )

    def __mul__(self, weight: float) -> LossOutput:
        if not isfinite(weight):
            raise ValueError("Loss weight must be finite.")
        return LossOutput(
            self.total * weight, self.losses, {name: value * weight for name, value in self.contributions.items()}
        )

    __rmul__ = __mul__

    def prefixed(self, prefix: str) -> LossOutput:
        """Namespace loss values before combining tasks; tracking adds stage and split."""
        return LossOutput(
            self.total,
            {f"{prefix}/{name}": value for name, value in self.losses.items()},
            {f"{prefix}/{name}": value for name, value in self.contributions.items()},
        )


@dataclass(frozen=True, slots=True)
class ModelOutput:
    outputs: Mapping[str, TensorTree] = field(default_factory=dict)
    features: Mapping[str, TensorTree] = field(default_factory=dict)
    losses: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepOutput:
    """Training requires loss; evaluation may report metrics without a loss."""

    loss: LossOutput | None
    predictions: Prediction = field(default_factory=dict)
    targets: Mapping[str, object] = field(default_factory=dict)


def validate_name(name: str, *, kind: str = "Component") -> None:
    if not name or name == "_" or any(char in name for char in "./") or name.strip() != name:
        raise ValueError(f"{kind} names must be nonblank, without dots, slashes or reserved '_'.")
