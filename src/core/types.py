"""Tensor trees and unbatched shapes shared by inputs and intermediate features."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

type TensorTree = Tensor | Mapping[str, TensorTree] | list[TensorTree] | tuple[TensorTree, ...] | None
type ShapeTree = TensorShape | Mapping[str, ShapeTree] | list[ShapeTree] | tuple[ShapeTree, ...] | None


def tree_map(operation: Callable[[Tensor], Tensor], tree: TensorTree) -> TensorTree:
    """Apply an operation to every tensor of a tree, preserving its structure and any other leaf as it is."""
    if isinstance(tree, Tensor):
        return operation(tree)
    if isinstance(tree, Mapping):
        return {name: tree_map(operation, value) for name, value in tree.items()}
    if isinstance(tree, list):
        return [tree_map(operation, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(operation, value) for value in tree)
    return tree


def require_tensor(value: object, *, name: str) -> Tensor:
    """Narrow a structured value where an operation requires a single tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}.")
    return value


CLASS_AXIS = 1
"""Where the class axis sits in a batched value: ``[B, C]`` and ``[B, C, H, W]`` alike.

Batch first, classes next is the layout every head produces and every loss reads, so its position is a
property of the tensors packages exchange rather than of any one package that acts on them.
"""


def drop_class_axis(values: Tensor) -> Tensor:
    """Drop a width-one class axis: ``[B, 1]`` becomes ``[B]``, ``[B, 1, H, W]`` becomes ``[B, H, W]``.

    One output per position means the class axis carries no information, while a target never has one;
    dropping it is what keeps a prediction comparable with what it is scored against.
    """
    return values.squeeze(CLASS_AXIS) if values.ndim > 1 and values.size(CLASS_AXIS) == 1 else values


def require_shape(value: ShapeTree, *, name: str) -> TensorShape:
    """Narrow a shape tree where an operation requires one tensor's shape."""
    if not isinstance(value, TensorShape):
        raise TypeError(f"{name} must be a single tensor shape, got {type(value).__name__}.")
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
