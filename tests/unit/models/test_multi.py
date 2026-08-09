"""``MultiEncoderBackbone``: one encoder per input, projected and stacked ``[B, N, D]``."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.core import Stream
from src.models import MultiEncoderBackbone
from src.models.registry import backbone_registry
from tests.support.fakes import FakeEncoder


def make_backbone(embedding_dim: int = 8) -> MultiEncoderBackbone:
    return MultiEncoderBackbone(
        encoders={"image": FakeEncoder("image", 4), "text": FakeEncoder("text", 6)},
        embedding_dim=embedding_dim,
    )


def make_inputs() -> dict[str, Tensor]:
    return {"image": torch.randn(3, 5), "text": torch.randn(3, 7)}


def test_stacks_projected_views_in_declaration_order() -> None:
    features = make_backbone()({**make_inputs()})

    assert features[Stream.EMBEDDINGS].shape == (3, 2, 8)


def test_embeddings_are_l2_normalized_by_default() -> None:
    embeddings = make_backbone()(make_inputs())[Stream.EMBEDDINGS]

    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_feature_dim_is_the_shared_embedding_dim() -> None:
    assert make_backbone(embedding_dim=16).feature_dim(Stream.EMBEDDINGS) == 16


def test_unknown_stream_is_rejected_by_name() -> None:
    with pytest.raises(LookupError, match=Stream.EMBEDDINGS):
        make_backbone().feature_dim(Stream.FEATURES)


def test_sub_encoders_and_projections_are_registered_submodules() -> None:
    backbone = make_backbone()

    module_names = dict(backbone.named_modules())
    assert "encoders.image" in module_names
    assert any(name.startswith("projections.text") for name in backbone.state_dict())


def test_registered_under_the_multi_key() -> None:
    assert "multi" in backbone_registry


def test_requires_at_least_two_encoders() -> None:
    with pytest.raises(ValueError, match="two"):
        MultiEncoderBackbone(encoders={"image": FakeEncoder("image", 4)}, embedding_dim=8)
