"""``MultiViewBackbone``: N views through one shared encoder, stacked ``[B, N, D]``."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from src.core import Backbone, Features, Stream
from src.models import MultiViewBackbone
from src.models.registry import backbone_registry


class MeanEncoder(Backbone):
    """Collapses an image to per-channel means — a deterministic stand-in encoder."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={Stream.FEATURES: inputs["image"].mean(dim=(2, 3))})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: 3}


def test_encodes_every_view_with_the_shared_inner_encoder() -> None:
    backbone = MultiViewBackbone(inner=MeanEncoder())
    views = torch.randn(4, 2, 3, 8, 8)

    features = backbone({"image": views})

    assert features[Stream.EMBEDDINGS].shape == (4, 2, 3)
    expected_first_view = views[:, 0].mean(dim=(2, 3))
    assert torch.allclose(features[Stream.EMBEDDINGS][:, 0], expected_first_view)


def test_optional_projection_maps_views_to_the_embedding_dim() -> None:
    backbone = MultiViewBackbone(inner=MeanEncoder(), embedding_dim=6)

    features = backbone({"image": torch.randn(2, 3, 3, 4, 4)})

    assert features[Stream.EMBEDDINGS].shape == (2, 3, 6)
    assert backbone.feature_dim(Stream.EMBEDDINGS) == 6


def test_feature_dim_defaults_to_the_inner_encoder_width() -> None:
    assert MultiViewBackbone(inner=MeanEncoder()).feature_dim(Stream.EMBEDDINGS) == 3


def test_unknown_stream_is_rejected_by_name() -> None:
    with pytest.raises(LookupError, match=Stream.EMBEDDINGS):
        MultiViewBackbone(inner=MeanEncoder()).feature_dim(Stream.FEATURES)


def test_the_inner_encoder_is_a_registered_submodule() -> None:
    backbone = MultiViewBackbone(inner=MeanEncoder())

    assert any(name == "inner" for name, _ in backbone.named_modules())


def test_registered_under_the_multiview_key() -> None:
    assert "multiview" in backbone_registry
