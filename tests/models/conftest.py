"""A backbone that publishes both shapes a head can read, and factories for the graphs model tests assemble."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Axis, Stream, TensorShape, TensorTree
from src.models import Backbone, CompositeModel, HeadConnection

POOLED_WIDTH, MAP_WIDTH, SIDE = 8, 4, 3

VECTOR = TensorShape(axes=(Axis.CHANNELS,), sizes=(POOLED_WIDTH,))
MAP = TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(MAP_WIDTH, None, None))


class Encoder(Backbone):
    """Two streams from one input: a pooled vector and a feature map, as timm and smp respectively publish."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, POOLED_WIDTH)

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: VECTOR, Stream.DECODER: MAP}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        images = inputs["image"]
        assert isinstance(images, Tensor)
        pooled = self.projection(images.flatten(1)[:, :2])
        return {Stream.POOLED: pooled, Stream.DECODER: pooled[:, :MAP_WIDTH, None, None].expand(-1, -1, SIDE, SIDE)}

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        """A pooled classifier of its own, as timm has one; nothing over the map, as a pooled encoder has none."""
        return nn.Linear(POOLED_WIDTH, out_features) if stream == Stream.POOLED else None


@pytest.fixture
def backbone() -> Encoder:
    return Encoder()


type CompositeFactory = Callable[..., CompositeModel]


@pytest.fixture
def make_composite(backbone: Encoder) -> CompositeFactory:
    """A composite over one linear head reading the pooled stream; keywords override any part."""

    def make(**overrides: Any) -> CompositeModel:
        heads: Mapping[str, HeadConnection] = {"label": HeadConnection(nn.Linear(POOLED_WIDTH, 2), input=Stream.POOLED)}
        parts: dict[str, Any] = {"backbone": backbone, "heads": heads}
        return CompositeModel(**{**parts, **overrides})

    return make


@pytest.fixture
def images() -> dict[str, Tensor]:
    """Two pictures whose pixels differ, so a gradient through them is observable."""
    return {"image": torch.arange(2 * 3 * SIDE * SIDE, dtype=torch.float32).reshape(2, 3, SIDE, SIDE)}
