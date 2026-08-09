"""``SmpBackbone``: an smp encoder+decoder exposing encoder and decoder streams."""

from __future__ import annotations

import pytest
import torch

from src.core import Stream
from src.models import SmpBackbone
from src.models.registry import backbone_registry


@pytest.fixture(scope="module")
def backbone() -> SmpBackbone:
    return SmpBackbone(arch="unet", encoder_name="resnet18", pretrained=False)


def test_exposes_encoder_and_decoder_streams(backbone: SmpBackbone) -> None:
    features = backbone({"image": torch.randn(2, 3, 64, 64)})

    assert features[Stream.ENCODER].shape == (2, 512, 2, 2)
    assert features[Stream.DECODER].shape == (2, 16, 64, 64)


def test_feature_dims_come_from_the_real_model(backbone: SmpBackbone) -> None:
    assert backbone.feature_dim(Stream.ENCODER) == 512
    assert backbone.feature_dim(Stream.DECODER) == 16


def test_unknown_stream_error_lists_both_streams(backbone: SmpBackbone) -> None:
    with pytest.raises(LookupError, match="encoder"):
        backbone.feature_dim(Stream.FEATURES)


def test_the_head_template_is_not_a_registered_submodule(backbone: SmpBackbone) -> None:
    """smp's own segmentation head is kept only as a cloning template."""
    assert not any("segmentation_head" in name for name in backbone.state_dict())


def test_native_decoder_head_maps_decoder_features_to_classes(backbone: SmpBackbone) -> None:
    head = backbone.native_head(Stream.DECODER, in_features=16, out_features=3)

    assert head is not None
    assert head(torch.randn(2, 16, 64, 64)).shape == (2, 3, 64, 64)


def test_native_encoder_head_pools_and_classifies(backbone: SmpBackbone) -> None:
    head = backbone.native_head(Stream.ENCODER, in_features=512, out_features=3)

    assert head is not None
    assert head(torch.randn(2, 512, 2, 2)).shape == (2, 3)


def test_native_head_is_none_for_unknown_streams(backbone: SmpBackbone) -> None:
    assert backbone.native_head(Stream.FEATURES, in_features=16, out_features=3) is None


def test_registered_under_the_smp_key() -> None:
    created = backbone_registry.create("smp", arch="unet", encoder_name="resnet18", pretrained=False)

    assert isinstance(created, SmpBackbone)


@pytest.mark.slow
def test_prefix_token_encoders_get_the_dino_norm_automatically(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One class covers DPT with ViT encoders: the norm patch applies itself, logged."""
    import logging

    with caplog.at_level(logging.INFO, logger="src.models.backbones.smp"):
        backbone = SmpBackbone(
            arch="dpt",
            encoder_name="tu-vit_small_patch16_224",
            pretrained=False,
            encoder_weights=None,
        )

    assert any("prefix tokens" in record.message for record in caplog.records)
    features = backbone({"image": torch.randn(1, 3, 224, 224)})
    assert Stream.ENCODER in features
    assert features[Stream.DECODER].dim() == 4


def test_plain_encoders_do_not_trigger_the_norm_patch(backbone: SmpBackbone, caplog: pytest.LogCaptureFixture) -> None:
    assert not any("prefix tokens" in record.message for record in caplog.records)
