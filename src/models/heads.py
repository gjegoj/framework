"""A head is any module the framework can build at two widths — the stream's and the task's output — while
every other argument comes from the declaration. That split is what lets a run swap one for another in a
line of YAML."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import ClassVar, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from src.core import DRAWN_AXIS, FEATURE_AXIS, Axis, Representation, as_children
from src.models.base import produced_by
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


@head_registry.register("mlp")
class Mlp(nn.Module):
    """Several projections of a pooled vector, with a nonlinearity between them.

    The head to declare where a run continues the tail of a larger one: a teacher trained with a wide
    head leaves a stack of layers, and a student whose backbone publishes the width that stack starts
    at carries that tail as its own head.

    What makes that arrangement carry anything is the nonlinearity. Projections with nothing between
    them multiply into a single matrix, so a frozen tail of such a stack is absorbed by whatever
    trainable layer sits beneath it — measured: the composition reaches every map the single layer
    reaches, and freezing it constrains nothing.

    GELU rather than a knob, because this is what the field builds the layer with — timm's own ``Mlp``
    defaults to it — and between two projections of a head the choice changes nothing a run is judged
    by. A knob with no real choice behind it is one more declaration to keep in step.

    Parameters:
        hidden_features: The width of each layer between the features read and the answer given, in the
            order they are read through. Left out, one layer as wide as what it reads, which is what
            this name carries elsewhere and what keeps every registered head buildable from its two
            widths alone. Declared empty it is refused: a head of one projection is ``linear``, and one
            idea spelled two ways is two statements free to drift.
    """

    reads_axes: ClassVar[tuple[str, ...]] = (Axis.CHANNELS,)

    def __init__(self, in_features: int, out_features: int, hidden_features: Sequence[int] | None = None) -> None:
        hidden = [in_features] if hidden_features is None else list(hidden_features)
        if not hidden:
            raise ValueError(
                "`hidden_features` is what this head holds and a single projection does not, and an "
                "empty list declares none of it; a head of one projection is `head: {name: linear}`."
            )
        if any(width < 1 for width in hidden):
            raise ValueError(f"Every layer of a head answers with at least one number; `hidden_features` was {hidden}.")
        super().__init__()
        through = [in_features, *hidden, out_features]
        layers: list[nn.Module] = []
        for reads, answers in pairwise(through):
            if layers:
                layers.append(nn.GELU())
            layers.append(nn.Linear(reads, answers))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features))


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

    @property
    def produces(self) -> Representation:
        """What the declaration answers with: a class space that grew has more rows, not another reading."""
        return produced_by(self.base)

    def forward(self, features: Tensor) -> Tensor:
        # The same axis carries classes for a flat output and for a dense one, which is why growing a
        # segmentation head and growing a classifier are one rule rather than two.
        return torch.cat((cast(Tensor, self.base(features)), cast(Tensor, self.novel(features))), dim=FEATURE_AXIS)


class StackedHeads(nn.Module):
    """One head per stream, their answers folded into the batch so that a sample's own stay adjacent.

    What a pairing produces: a picture read by one tower and its caption by another, projected into one
    space by a head apiece and handed on as ordinary rows. ``head: {name: linear, stream: [image_pooled,
    text_pooled]}`` is the whole declaration, and the heads are the declared one built once per stream,
    at the width each of them publishes — which is why the towers need not be the same width and why
    every head this framework has works in a pairing without knowing one exists.

    Not in the head registry: nobody declares one. It is assembled by the builder that knows how many
    streams the declaration named, exactly as ``ExpandedHead`` is assembled around the rows a file
    carried.

    Folded rather than kept apart on an axis of their own — the decision drawn views already made, and
    the same fact underneath it: what a task declares it produces is what *one* answer is, so a batch
    answering twice per sample holds more rows rather than deeper ones. Everything downstream reads a
    pairing exactly as it reads a pair of views, the objective over them included.
    """

    def __init__(self, heads: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.heads = as_children(heads, label="stream")

    @property
    def produces(self) -> Representation:
        """One declaration built once per stream, so what any of them answers with is what all of them do."""
        return produced_by(next(iter(self.heads.values())))

    def forward(self, *features: Tensor) -> Tensor:
        answered = [cast(Tensor, head(feature)) for head, feature in zip(self.heads.values(), features, strict=True)]
        return torch.stack(answered, dim=DRAWN_AXIS).flatten(0, DRAWN_AXIS)
