"""A head is any module the framework can build at two widths — the stream's and the task's output — while
every other argument comes from the declaration. That split is what lets a run swap one for another in a
line of YAML."""

from __future__ import annotations

from typing import ClassVar, cast

from torch import Tensor, nn

from src.core import Axis
from src.models.registry import head_registry


@head_registry.register("linear")
class LinearHead(nn.Module):
    """One projection of a pooled vector — the default for a whole-sample output."""

    reads: ClassVar[tuple[str, ...]] = (Axis.CHANNELS,)

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

    reads: ClassVar[tuple[str, ...]] = (Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH)

    def __init__(self, in_features: int, out_features: int, kernel_size: int = 1) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_features, out_features, kernel_size, padding=kernel_size // 2)

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.projection(features))
