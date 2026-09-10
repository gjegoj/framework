"""Tensor trees and unbatched shapes shared by inputs and intermediate features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

type TensorTree = Tensor | Mapping[str, TensorTree] | list[TensorTree] | tuple[TensorTree, ...] | None
type ShapeTree = TensorShape | Mapping[str, ShapeTree] | list[ShapeTree] | tuple[ShapeTree, ...] | None


def require_tensor(value: TensorTree, *, name: str) -> Tensor:
    """Narrow a structured value where an operation requires a single tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}.")
    return value


@dataclass(frozen=True, slots=True)
class TensorShape:
    """Axes exclude the batch dimension; custom axis names remain valid strings."""

    axes: tuple[str, ...]
    sizes: tuple[int | None, ...]
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.sizes) or len(set(self.axes)) != len(self.axes):
            raise ValueError("Axes and sizes must match, with distinct axis names.")
        if any(not axis or axis.strip() != axis for axis in self.axes):
            raise ValueError("Axis names must be nonblank.")
        if any(size is not None and (isinstance(size, bool) or size < 0) for size in self.sizes):
            raise ValueError("Sizes must be nonnegative or None.")

    def size(self, axis: str) -> int | None:
        """Fail explicitly when a head requests an unavailable semantic axis."""
        if axis not in self.axes:
            raise ValueError(f"Required axis {axis!r} is missing from {self.axes}.")
        return self.sizes[self.axes.index(axis)]
