"""timm as a backbone: any of its models as a pooled-feature encoder, with its own classifier on offer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import timm
from torch import Tensor, nn

from src.core import Axis, Modality, Stream, TensorShape, TensorTree, require_tensor
from src.models.base import Backbone
from src.models.registry import backbone_registry


@backbone_registry.register("timm")
class TimmBackbone(Backbone):
    """Built with ``num_classes=0``, so the model ends at its pooling and publishes one ``[B, C]`` stream.

    Args:
        model_name: timm's id, e.g. ``"resnet18"``.
        pretrained: Load the pretrained weights timm hosts.
        input_name: Which of the batch's inputs to encode.
        **options: Forwarded to ``timm.create_model`` (``in_chans``, ``drop_rate``, ...).
    """

    def __init__(
        self, model_name: str, pretrained: bool = True, input_name: str = Modality.IMAGE, **options: Any
    ) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **options)
        self.input_name = input_name
        self.width = int(cast(int, self.model.num_features))

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,))}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        pooled = cast(Tensor, self.model(require_tensor(inputs[self.input_name], name=self.input_name)))
        if pooled.ndim != 2:
            raise ValueError(
                f"{self.input_name!r} encoded to {pooled.ndim} axes; this adapter publishes a pooled [batch, "
                "channels] stream. A model whose features stay spatial belongs behind an encoder-decoder adapter."
            )
        return {Stream.POOLED: pooled}

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        """timm's own classifier for this family, at the number of classes the task needs."""
        if stream != Stream.POOLED:
            return None
        from timm.layers import create_classifier

        _, classifier = create_classifier(self.width, out_features)
        return cast(nn.Module, classifier)
