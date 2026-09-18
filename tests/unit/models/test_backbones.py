"""Library adapters: what timm and smp publish, and the head each of them brings."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Axis, Stream, TensorShape
from src.models import Backbone
from src.models.backbones import SmpBackbone, TimmBackbone
from src.models.build import build_model
from src.models.registry import backbone_registry

PICTURES = {"image": torch.zeros(2, 3, 64, 64)}


@pytest.mark.parametrize("name", list(backbone_registry))
def test_every_registered_backbone_is_one(name: str) -> None:
    assert issubclass(backbone_registry.get(name), Backbone)


@pytest.fixture(scope="module")
def timm_backbone() -> TimmBackbone:
    return TimmBackbone("resnet18", pretrained=False)


@pytest.fixture(scope="module")
def smp_backbone() -> SmpBackbone:
    return SmpBackbone(arch="unet", encoder_name="resnet18", pretrained=False)


class TestTimm:
    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("resnet18", id="a head that classifies the features it is handed"),
            pytest.param("mobilenetv4_conv_small", id="a head that widens them through a hidden layer"),
            pytest.param("poolformerv2_s12", id="a family that publishes no width for its head"),
        ],
    )
    def test_the_width_it_declares_is_the_width_its_model_returns(self, model_name: str) -> None:
        """Otherwise a head is built against one number and fed another, and the run dies on its first
        matmul with torch's word about two shapes, naming neither the model nor the line that chose it.

        Measured over 32 families on timm 1.0.28: ``num_features`` is the width *before* the head, and
        ten of them — every mobilenet, ghostnet, lcnet, repghostnet, hardcorenas, vgg — end wider than
        it, vgg11 at 4096 against a stated 512. These three are the three readings timm has: a head
        that classifies what it is handed, one that widens it first, and a family publishing no width
        for its head at all.
        """
        backbone = TimmBackbone(model_name, pretrained=False)

        pooled = backbone(PICTURES)[Stream.POOLED]

        assert pooled.shape[1] == backbone.feature_shapes[Stream.POOLED].size(Axis.CHANNELS)

    def test_publishes_one_pooled_vector_and_encodes_into_it(self, timm_backbone: TimmBackbone) -> None:
        shapes = timm_backbone.feature_shapes

        features = timm_backbone(PICTURES)

        assert set(shapes) == {Stream.POOLED} and shapes[Stream.POOLED].axes == (Axis.CHANNELS,)
        assert features[Stream.POOLED].shape == (2, shapes[Stream.POOLED].size(Axis.CHANNELS))

    def test_brings_its_own_classifier_over_the_pooled_stream(self, timm_backbone: TimmBackbone) -> None:
        native = timm_backbone.native_head(Stream.POOLED, out_features=7)
        assert native is not None

        assert native(timm_backbone(PICTURES)[Stream.POOLED]).shape == (2, 7)

    def test_offers_no_head_for_a_stream_it_does_not_publish(self, timm_backbone: TimmBackbone) -> None:
        assert timm_backbone.native_head(Stream.DECODER, out_features=7) is None


class TestSmp:
    def test_publishes_an_encoder_map_and_a_decoder_map(self, smp_backbone: SmpBackbone) -> None:
        shapes = smp_backbone.feature_shapes

        features = smp_backbone(PICTURES)

        assert set(shapes) == {Stream.ENCODER, Stream.DECODER}
        for stream, feature in features.items():
            assert shapes[stream].axes == (Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH)
            assert feature.shape[1] == shapes[stream].size(Axis.CHANNELS) and feature.ndim == 4
        assert features[Stream.DECODER].shape[-2:] == (64, 64)  # the decoder returns the image's own size

    @pytest.mark.parametrize(
        ("stream", "spatial"),
        [(Stream.DECODER, True), (Stream.ENCODER, False)],
        ids=["the segmentation head over the decoder", "a pooling classifier over the encoder"],
    )
    def test_brings_a_head_for_each_stream_shaped_as_that_stream_deserves(
        self, smp_backbone: SmpBackbone, stream: str, spatial: bool
    ) -> None:
        native = smp_backbone.native_head(stream, out_features=5)
        assert native is not None

        output = native(smp_backbone(PICTURES)[stream])

        assert output.shape[:2] == (2, 5) and (output.ndim == 4) is spatial

    def test_its_own_head_never_runs_in_the_forward_pass(self, smp_backbone: SmpBackbone) -> None:
        """The template is kept for `native_head` alone, so its weights reach no checkpoint or export."""
        assert not any("segmentation_head" in name for name, _ in smp_backbone.named_parameters())


class TestDpt:
    """A ViT encoder under smp's DPT decoder: the one family whose encoder hands its decoder prefix tokens."""

    @pytest.fixture(scope="class")
    def dpt(self) -> SmpBackbone:
        return SmpBackbone(
            arch="dpt",
            encoder_name="tu-vit_small_patch16_224.augreg_in21k",
            pretrained=False,
            dynamic_img_size=False,
            img_size=[64, 64],
        )

    @pytest.mark.slow
    def test_publishes_the_same_two_streams_and_encodes_through_its_prefix_tokens(self, dpt: SmpBackbone) -> None:
        """The branch is chosen by asking the encoder, so a new architecture of this shape needs no list entry."""
        features = dpt(PICTURES)

        assert dpt.carries_prefix_tokens
        assert set(features) == {Stream.ENCODER, Stream.DECODER}
        for stream, feature in features.items():
            assert feature.ndim == 4 and feature.shape[1] == dpt.feature_shapes[stream].size(Axis.CHANNELS)

    @pytest.mark.slow
    def test_reassembles_intermediate_features_through_the_encoders_final_norm(self, dpt: SmpBackbone) -> None:
        """The DINO recipe: smp reads intermediates without the ViT's last LayerNorm unless its hook is replaced."""
        encoded = dpt({"image": torch.randn(2, 3, 64, 64)})[Stream.ENCODER]

        assert float(encoded.mean()) == pytest.approx(0.0, abs=1e-3)
        assert float(encoded.std()) == pytest.approx(1.0, abs=1e-2)

    @pytest.mark.slow
    def test_brings_its_own_dense_head(self, dpt: SmpBackbone) -> None:
        native = dpt.native_head(Stream.DECODER, out_features=5)
        assert native is not None

        assert native(dpt(PICTURES)[Stream.DECODER]).shape[:2] == (2, 5)


@pytest.mark.parametrize(
    ("backbone", "head", "stream", "shape"),
    [
        pytest.param(
            {"name": "timm", "model_name": "resnet18", "pretrained": False},
            "native",
            Stream.POOLED,
            (2, 4),
            id="a timm classifier",
        ),
        pytest.param(
            {"name": "smp", "encoder_name": "resnet18", "pretrained": False},
            "conv",
            Stream.DECODER,
            (2, 4, 64, 64),
            id="a conv head on an smp decoder",
        ),
    ],
)
def test_a_declared_backbone_composes_into_a_running_model(
    backbone: dict[str, Any], head: str, stream: str, shape: tuple[int, ...]
) -> None:
    from src.config import ComponentConfig, HeadConfig, ModelConfig

    axes = (Axis.CLASSES,) if len(shape) == 2 else (Axis.CLASSES, Axis.HEIGHT, Axis.WIDTH)
    output_shape = TensorShape(axes=axes, sizes=(4, *(None,) * (len(axes) - 1)))
    declared = ModelConfig(name="composite", backbone=ComponentConfig(**backbone))

    model = build_model(declared, {"t": HeadConfig(name=head, stream=stream)}, {"t": output_shape})

    assert model(PICTURES).outputs["t"].shape == shape
