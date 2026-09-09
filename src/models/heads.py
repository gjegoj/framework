"""Built-in heads for the composite family: what a task's features become its outputs through."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from src.core.ports import one_stream
from src.models.registry import head_registry

if TYPE_CHECKING:
    from collections.abc import Mapping


@head_registry.register("linear")
class LinearHead(nn.Module):
    """A single linear projection — the default head for a global output."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._projection = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        features = one_stream(features, head=type(self).__name__)
        # nn.Module.__call__ erases the return type to Any; pin it back.
        return cast("Tensor", self._projection(features))


@head_registry.register("conv")
class ConvHead(nn.Module):
    """Channel projection over spatial features — the default DENSE head.

    Maps ``[B, in, H, W]`` to ``[B, out, H, W]``; wider kernels keep the
    spatial size via same-padding.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int = 1) -> None:
        super().__init__()
        self._projection = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        features = one_stream(features, head=type(self).__name__)
        # nn.Module.__call__ erases the return type to Any; pin it back.
        return cast("Tensor", self._projection(features))


@head_registry.register("cosine")
class CosineHead(nn.Module):
    """Cosine similarities to learnable class prototypes — the angular-margin classifier.

    Normalizes the feature and every prototype, so each logit is ``cos(θ)``. The margin
    belongs to the loss (``arcface``), never to the head: argmax over cosines is the
    predicted class, and CosFace later is a loss swap.

    Parameters:
        in_features (int): Width of the stream the task reads.
        out_features (int): One prototype per class.
        embedding_dim (int | None): Project features to this width (bias-free) before
            comparing — the deployable embedding. ``None`` compares backbone features directly.
    """

    def __init__(self, in_features: int, out_features: int, embedding_dim: int | None = None) -> None:
        super().__init__()
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError(f"CosineHead embedding_dim must be positive, got {embedding_dim}.")
        self._projection = nn.Linear(in_features, embedding_dim, bias=False) if embedding_dim is not None else None
        self.prototypes = nn.Parameter(torch.empty(out_features, embedding_dim or in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        features = one_stream(features, head=type(self).__name__)
        embedding = self._projection(features) if self._projection is not None else features
        return functional.linear(functional.normalize(embedding, dim=1), functional.normalize(self.prototypes, dim=1))


class ExpandedHead(nn.Module):
    """A classifier whose class space grew: trained ``base`` rows beside a fresh ``novel`` block.

    Two modules instead of one wider matrix on purpose: ``requires_grad``
    lives on whole tensors, and a gradient mask would still let AdamW's
    decoupled decay move the "frozen" rows. With the boundary as a submodule,
    the freeze callback, the optimizer, and EMA all work unchanged
    (``modules: [heads.<task>.base]``). The submodule names are that
    contract. Not registry-listed — the timm adapter assembles it around a
    transplanted checkpoint classifier.
    """

    def __init__(self, base: nn.Module, novel: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.novel = novel

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        features = one_stream(features, head=type(self).__name__)
        # dim=1 is the class axis of batch-first logits — [B, C] and [B, C, H, W] alike.
        return torch.cat((self.base(features), self.novel(features)), dim=1)


class DetectHead(nn.Module):
    """A detection head over a pyramid: the levels in, one raw tensor out.

    Wraps ultralytics' ``Detect`` without importing it — the backbone that owns the graph
    builds and hands it over. ``[B, 4·reg_max + nc, A]`` in train and eval alike: ultralytics
    returns a dict in train and ``(decoded, dict)`` in eval, and this reads the dict from
    either. Decoding is a separate step, not a mode.

    Parameters:
        detect (nn.Module): The wrapped ``Detect``.
        streams (tuple[str, ...]): The pyramid levels, in the order the head reads them.
        strides (tuple[int, ...]): Stride per level.
        reg_max (int): Bins per box side in the DFL distribution.
    """

    def __init__(self, detect: nn.Module, *, streams: tuple[str, ...], strides: tuple[int, ...], reg_max: int) -> None:
        super().__init__()
        self.detect = detect
        self.streams = streams
        self.strides = strides
        self.reg_max = reg_max
        module: Any = detect  # nn.Module types every attribute as Tensor | Module; nc is an int
        self.num_classes = int(module.nc)

    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(features, Tensor):
            raise TypeError(f"DetectHead reads the pyramid {', '.join(self.streams)}, but was handed one stream.")
        raw = self.detect([features[name] for name in self.streams])
        predictions = raw[1] if isinstance(raw, tuple) else raw
        return torch.cat([predictions["boxes"], predictions["scores"]], dim=1)
