"""Unit tests for the model layer (offline, pretrained=False)."""

import pytest
import torch

from src.core.entities import HeadSpec
from src.core.keys import DECODER, ENCODER_LAST, IMAGE, POOLED
from src.models import CompositeModel, build_composite_model
from src.models.backbones import EmbeddingBackbone, SmpBackbone, TimmBackbone
from src.models.registry import backbones, head_builders


def _images(batch: int = 2) -> dict[str, torch.Tensor]:
    return {IMAGE: torch.randn(batch, 3, 32, 32)}


class TestTimmBackbone:
    def test_pooled_stream_and_dim(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        features = backbone(_images(2))
        assert features[POOLED].shape == (2, 512)
        assert backbone.feature_dim(POOLED) == 512

    def test_unknown_stream_raises(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        with pytest.raises(KeyError, match="exposes only 'pooled'"):
            backbone.feature_dim("decoder")


class TestEmbeddingBackbone:
    def test_passthrough_stream_and_dim(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=16)
        vectors = {IMAGE: torch.randn(2, 16)}
        features = backbone(vectors)
        assert features[POOLED].shape == (2, 16)
        assert backbone.feature_dim(POOLED) == 16

    def test_reads_configured_input_key(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=8, input_key="embedding")
        features = backbone({"embedding": torch.randn(3, 8)})
        assert features[POOLED].shape == (3, 8)

    def test_unknown_stream_raises(self) -> None:
        backbone = EmbeddingBackbone(embedding_dim=16)
        with pytest.raises(KeyError, match="exposes only 'pooled'"):
            backbone.feature_dim("decoder")

    def test_builds_from_config_ignoring_name_and_pretrained(self) -> None:
        from src.composition.wiring import build_backbone
        from src.config.schema import BackboneConfig

        cfg = BackboneConfig(kind="embedding", name="identity", embedding_dim=8, input_key="embedding")
        backbone = build_backbone(cfg)
        assert isinstance(backbone, EmbeddingBackbone)
        assert backbone.feature_dim(POOLED) == 8


class TestConvHead:
    def test_preserves_spatial_dims_maps_channels(self) -> None:
        head = head_builders.create("conv", in_features=16, out_features=4)
        features = torch.randn(2, 16, 24, 24)  # [B, D, H, W]
        logits = head(features)
        assert logits.shape == (2, 4, 24, 24)  # [B, C, H, W]


class TestSmpBackbone:
    def test_exposes_encoder_last_and_decoder_streams(self) -> None:
        backbone = SmpBackbone(name="unet", encoder_name="resnet18", pretrained=False)
        features = backbone(_images(2))  # 32x32 input (divisible by 32)
        assert set(features.streams) == {ENCODER_LAST, DECODER}
        assert features[DECODER].shape[0] == 2
        assert features[DECODER].shape[2:] == (32, 32)  # full input resolution
        assert features[DECODER].shape[1] == backbone.feature_dim(DECODER)
        enc = features[ENCODER_LAST]
        assert enc.ndim == 4, "encoder_last must be [B, D, H, W]"
        assert enc.shape[0] == 2
        assert enc.shape[1] == backbone.feature_dim(ENCODER_LAST)

    def test_unknown_stream_raises_with_listing(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        with pytest.raises(KeyError, match="encoder_last"):
            backbone.feature_dim("tokens")

    def test_native_head_encoder_last_returns_classification_head(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        in_features = backbone.feature_dim(ENCODER_LAST)
        head = backbone.native_head(ENCODER_LAST, in_features, out_features=5)
        assert head is not None
        x = torch.randn(2, in_features, 4, 4)
        out = head(x)
        assert out.shape == (2, 5)

    def test_native_head_unknown_key_returns_none(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        assert backbone.native_head("pooled", 512, 3) is None

    def test_segmentation_model_produces_per_pixel_logits(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        specs = {"mask": HeadSpec(kind="conv", out_features=4, feature_key=DECODER)}
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["mask"].shape == (2, 4, 32, 32)  # [B, C, H, W]

    def test_multitask_segmentation_and_classification(self) -> None:
        backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
        specs = {
            "mask": HeadSpec(kind="conv", out_features=3, feature_key=DECODER, prefer_native=True),
            "label": HeadSpec(kind="linear", out_features=5, feature_key=ENCODER_LAST, prefer_native=True),
        }
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["mask"].shape == (2, 3, 32, 32)
        assert output.task_logits["label"].shape == (2, 5)


class TestRegistries:
    def test_builtins_registered(self) -> None:
        assert "timm" in backbones
        assert "smp" in backbones
        assert "linear" in head_builders
        assert "conv" in head_builders


class TestCompositeModel:
    def test_single_head_forward(self) -> None:
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        specs = {"label": HeadSpec(kind="linear", out_features=3, feature_key=POOLED)}
        model = build_composite_model(backbone, specs)

        assert isinstance(model, CompositeModel)
        output = model(_images(4))
        assert set(output.task_logits) == {"label"}
        assert output.task_logits["label"].shape == (4, 3)
        assert output.features[POOLED].shape == (4, 512)

    def test_multi_head_shares_backbone(self) -> None:
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        specs = {
            "species": HeadSpec(kind="linear", out_features=3),
            "age": HeadSpec(kind="linear", out_features=1),
        }
        model = build_composite_model(backbone, specs)
        output = model(_images(2))
        assert output.task_logits["species"].shape == (2, 3)
        assert output.task_logits["age"].shape == (2, 1)
        # One shared backbone instance, two heads.
        assert len(model.heads) == 2
