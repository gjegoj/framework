"""timm backbone adapter: a feature extractor producing a pooled global vector."""

from __future__ import annotations

from typing import Any, cast

import timm
from torch import Tensor

from src.core.entities import FeatureBundle
from src.core.keys import IMAGE, POOLED
from src.core.ports import Backbone
from src.models.registry import backbones


@backbones.register("timm")
class TimmBackbone(Backbone):
    """Wraps a timm model as a backbone exposing a single ``pooled`` stream.

    The timm model is created with ``num_classes=0`` so its forward returns the
    pooled feature vector ``[B, num_features]``.

    Parameters:
        name (str): timm model name (e.g. ``"resnet18"``).
        pretrained (bool): Load pretrained weights (requires network).
        input_key (str): Which input modality to encode (defaults to ``image``).
        **kwargs (object): Extra keyword args forwarded to ``timm.create_model``.
    """

    def __init__(self, name: str, pretrained: bool = True, input_key: str = IMAGE, **kwargs: Any) -> None:
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
        self._num_features = cast(int, self.model.num_features)
        self._input_key = input_key

    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        pooled = self.model(inputs[self._input_key])
        return FeatureBundle(streams={POOLED: pooled})

    def feature_dim(self, key: str) -> int:
        if key != POOLED:
            raise KeyError(f"TimmBackbone exposes only {POOLED!r}, requested {key!r}.")
        return self._num_features

    def native_head(self, feature_key: str, in_features: int, out_features: int) -> Any:
        if feature_key != POOLED:
            return None
        from timm.layers import create_classifier

        _, classifier = create_classifier(in_features, out_features)
        return classifier
