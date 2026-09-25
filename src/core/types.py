"""Tensor trees and unbatched shapes shared by inputs and intermediate features, and the unit a size is declared in."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import cast

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


def tensors_in(tree: TensorTree) -> Iterator[Tensor]:
    """Every tensor a tree holds, in the order it holds them, whatever it holds them inside.

    The reading half of ``tree_map``: a question about the tensors of a structure, rather than a new
    structure built from them. One home, because a second walk would be free to disagree about what
    counts as a leaf — which is the whole of what these two functions know.
    """
    if isinstance(tree, Tensor):
        yield tree
    elif isinstance(tree, Mapping):
        for value in tree.values():
            yield from tensors_in(value)
    elif isinstance(tree, list | tuple):
        for value in tree:
            yield from tensors_in(value)


def require_tensor(value: object, *, name: str) -> Tensor:
    """Narrow a structured value where an operation requires a single tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}.")
    return value


def require_named_tensors(value: object, *, name: str) -> Mapping[str, Tensor]:
    """Narrow a structured value where an operation requires several tensors that answer to names.

    The other half of ``require_tensor``: one input may be a tree rather than a tensor — a sentence
    reaches a model as ids beside the mask saying which of them are words — and whoever reads such an
    input needs the names, not a single value. Flat, because a flat mapping is what the libraries that
    take them are called with.
    """
    if not isinstance(value, Mapping) or not all(isinstance(one, Tensor) for one in value.values()):
        arrived = f"a tree of {', '.join(sorted(value))}" if isinstance(value, Mapping) else type(value).__name__
        raise TypeError(f"{name} must be named tensors, got {arrived}.")
    return cast("Mapping[str, Tensor]", value)


FEATURE_AXIS = 1
"""Where the values a head makes per position sit in a batched tensor: ``[B, C]`` and ``[B, C, H, W]``.

Batch first, then whatever the head produces, which is the layout every head writes and every loss
reads — so its position belongs to the tensors packages exchange rather than to any one of them.
``Task.out_features`` is how many values sit here; ``Axis.CLASSES`` and ``Axis.EMBEDDING`` are the
shape's own words for what they mean.
"""

DRAWN_AXIS = 1
"""Where a sample's several answers sit before they are folded into the batch, and its several draws too.

One axis rather than a name in any declared shape, and it rides beside the batch axis — the one
``TensorShape`` deliberately does not carry either. What a task declares it produces is what *one*
answer is, and what an input *is* stays what one view is, which is also what a deployed artifact takes
— so a run that draws a picture twice, or pairs it with its caption, has changed how many rows a batch
holds rather than how deep each of them is. Both folds leave a sample's own answers adjacent, which is
the order an objective comparing them relies on, and both read this one name.
"""

BYTES_PER_GIB = 1024**3
"""Binary, as a size declared in GiB is read. In the core because more than one package declares one, and a
package does not import another to learn a unit."""


def drop_feature_axis(values: Tensor) -> Tensor:
    """Drop a width-one feature axis: ``[B, 1]`` becomes ``[B]``, ``[B, 1, H, W]`` becomes ``[B, H, W]``.

    One value per position means the axis carries no information, while a target never has one;
    dropping it is what keeps a prediction comparable with what it is scored against.
    """
    return values.squeeze(FEATURE_AXIS) if values.ndim > 1 and values.size(FEATURE_AXIS) == 1 else values


@dataclass(frozen=True, slots=True)
class TensorShape:
    """Axes exclude the batch dimension; custom axis names remain valid strings."""

    axes: tuple[str, ...]
    sizes: tuple[int | None, ...]

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
