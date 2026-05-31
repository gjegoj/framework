"""Unit tests for the model layer (offline, pretrained=False)."""

import pytest
import torch

from src.core.entities import HeadSpec
from src.core.keys import IMAGE, POOLED
from src.models import CompositeModel, build_composite_model
from src.models.backbones import TimmBackbone
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


class TestRegistries:
    def test_builtins_registered(self) -> None:
        assert "timm" in backbones
        assert "linear" in head_builders


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
