"""Adapter for timm classifiers used as pooled feature extractors."""

from collections.abc import Mapping
from typing import cast

from torch import Tensor

from src.core import Axis, Modality, Stream, TensorShape, TensorTree
from src.core.types import require_tensor
from src.models.backbones.base import Backbone


class TimmBackbone(Backbone):
    def __init__(self, model_name: str, *, pretrained: bool = False, input_name: str = Modality.IMAGE) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.input_name = input_name

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(cast(int, self.encoder.num_features),))}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        images = require_tensor(inputs[self.input_name], name=self.input_name)
        features = cast(Tensor, self.encoder(images))
        if features.ndim != 2:
            raise ValueError("This adapter requires pooled [batch, channels] features.")
        return {Stream.POOLED: features}
