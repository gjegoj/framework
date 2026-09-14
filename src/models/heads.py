"""A head is any module the framework can build at two widths — the stream's and the task's output — while
every other argument comes from the declaration. That split is what lets a run swap one for another in a
line of YAML."""

from __future__ import annotations

from typing import ClassVar, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from src.core import FEATURE_AXIS, Axis, Representation
from src.models.registry import head_registry


@head_registry.register("linear")
class LinearHead(nn.Module):
    """One projection of a pooled vector — the default for a whole-sample output."""

    reads_axes: ClassVar[tuple[str, ...]] = (Axis.CHANNELS,)

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.projection(features))


@head_registry.register("conv")
class ConvHead(nn.Module):
    """A channel projection over a feature map — the default for a dense output.

    ``[B, in, H, W]`` becomes ``[B, out, H, W]``; a wider kernel keeps the size through same-padding.
    """

    reads_axes: ClassVar[tuple[str, ...]] = (Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH)

    def __init__(self, in_features: int, out_features: int, kernel_size: int = 1) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_features, out_features, kernel_size, padding=kernel_size // 2)

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.projection(features))


@head_registry.register("cosine")
class CosineHead(nn.Module):
    """The angle between a pooled feature and one prototype per class, both read as directions alone.

    Declared under an angular objective, which needs a cosine to add its margin to — a plain projection
    would saturate and the margin would mean nothing. Its prototypes are parameters of the *network*,
    so they are exported with it and the artifact classifies; ``kind: metric_learning`` is the same
    arrangement made the other way.

    ``embedding_dim`` narrows the feature before the angles are taken, for a backbone far wider than
    the space the identities need; left out, the stream's own width is that space.
    """

    reads_axes: ClassVar[tuple[str, ...]] = (Axis.CHANNELS,)
    produces: ClassVar[Representation] = Representation.COSINES

    def __init__(self, in_features: int, out_features: int, embedding_dim: int | None = None) -> None:
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError(f"'embedding_dim' is the width the angles are taken in; it was {embedding_dim}.")
        super().__init__()
        # No bias: a shift would move the origin the angles are measured from, which is the one thing
        # a direction cannot carry.
        self.projection = nn.Identity() if embedding_dim is None else nn.Linear(in_features, embedding_dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(out_features, embedding_dim or in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features: Tensor) -> Tensor:
        projected = cast(Tensor, self.projection(features))
        return normalize(projected, dim=FEATURE_AXIS) @ normalize(self.prototypes, dim=-1).T


class ExpandedHead(nn.Module):
    """A class space that grew: rows carried from a file beside fresh ones for the classes added since.

    Two submodules rather than one wider projection, and that is the whole design. ``requires_grad``
    lives on whole tensors, so one matrix would make "hold what was learned still" and "hold the new
    classes still" the same instruction; with the boundary as a path, ``freeze``, the optimizer's
    per-task groups and the averaging callback all address them apart, unchanged — ``modules:
    [heads.<task>.base]`` is that contract, and the submodule names are what a declaration writes.

    Not in the head registry: nobody declares one. It is assembled around a classifier a weight file
    carried, by the builder that knows both how many rows arrived and how many the task asks for.
    """

    def __init__(self, base: nn.Module, novel: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.novel = novel

    def forward(self, features: Tensor) -> Tensor:
        # The same axis carries classes for a flat output and for a dense one, which is why growing a
        # segmentation head and growing a classifier are one rule rather than two.
        return torch.cat((cast(Tensor, self.base(features)), cast(Tensor, self.novel(features))), dim=FEATURE_AXIS)
