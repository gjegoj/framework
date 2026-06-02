"""SmpBackbone: a segmentation-models-pytorch encoder+decoder as a feature source.

We keep smp's encoder and decoder but drop its built-in segmentation head, so the
backbone produces a dense ``decoder`` feature map ``[B, D, H, W]`` that our own
``ConvHead`` turns into class logits — preserving the framework's
backbone→FeatureBundle→head split (and per-head LR / multitask on one backbone).

It also exposes a ``pooled`` stream (global-pooled deepest encoder feature) so a
classification head can share the same backbone as a segmentation head.
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
from torch import Tensor

from src.core.entities import FeatureBundle
from src.core.keys import DECODER, IMAGE, POOLED
from src.core.ports import Backbone
from src.models.registry import backbones


@backbones.register("smp")
class SmpBackbone(Backbone):
    """Encoder+decoder from smp, exposing ``decoder`` and ``pooled`` streams.

    Parameters:
        name (str): smp architecture name (e.g. ``"unet"``, ``"unetplusplus"``).
        encoder_name (str): Encoder backbone (e.g. ``"resnet18"``).
        pretrained (bool): Load ImageNet encoder weights when ``True``.
        **kwargs (Any): Extra args forwarded to ``smp.create_model``.
    """

    def __init__(
        self,
        name: str = "unet",
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        model = smp.create_model(
            arch=name,
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=1,  # unused: we drop smp's segmentation head
            **kwargs,
        )
        self._encoder = model.encoder
        self._decoder = model.decoder
        self._pooled_dim = int(model.encoder.out_channels[-1])
        self._decoder_dim = int(model.segmentation_head[0].in_channels)

    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        features = self._encoder(inputs[IMAGE])
        decoder = self._decoder(features)
        pooled = features[-1].mean(dim=(2, 3))  # global average pool of deepest feature
        return FeatureBundle({DECODER: decoder, POOLED: pooled})

    def feature_dim(self, key: str) -> int:
        if key == DECODER:
            return self._decoder_dim
        if key == POOLED:
            return self._pooled_dim
        raise KeyError(f"SmpBackbone exposes 'decoder'/'pooled', not {key!r}.")
