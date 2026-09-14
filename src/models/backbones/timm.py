"""timm as a backbone: any of its models as a pooled-feature encoder, with its own classifier on offer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import timm
from torch import Tensor, nn

from src.core import Axis, Modality, Stream, TensorShape, TensorTree, require_tensor
from src.models.base import Backbone, required_input
from src.models.registry import backbone_registry
from src.models.weights import start_from

INSIDE = "model."
"""Where this adapter keeps timm's own graph, which is the one namespace a weight file writes.

Measured on timm 1.0: a resnet18 file and this backbone share none of their 120 names, and all 120
once this prefix is accounted for. A declaration naming a file says nothing about it.
"""


@backbone_registry.register("timm")
class TimmBackbone(Backbone):
    """Built with ``num_classes=0``, so the model ends at its pooling and publishes one ``[B, C]`` stream.

    Args:
        model_name: timm's id, e.g. ``"resnet18"``.
        pretrained: Load the pretrained weights timm hosts.
        input_name: Which of the batch's inputs to encode.
        checkpoint_path: Weights of this architecture from elsewhere — another run, a pretraining
            script, a hub file — put in after timm has built the graph. The classifier such a file
            carries is held back rather than loaded, because this graph is built without one.
        **options: Forwarded to ``timm.create_model`` (``in_chans``, ``drop_rate``, ...).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        input_name: str = Modality.IMAGE,
        *,
        checkpoint_path: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        # Named in the signature rather than left to `**options`: timm's own `create_model` takes a
        # `checkpoint_path` too, and measured, it refuses a trained file against `num_classes=0` with
        # torch's bare word about `fc.weight` — naming neither the declaration nor the way out.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **options)
        self.input_name = input_name
        self.width = int(cast(int, self.model.num_features))
        if checkpoint_path is not None:
            self.carried_head = start_from(self, checkpoint_path, inside=INSIDE, aside=_classifier_prefixes(self.model))

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,))}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        image = required_input(inputs, self.input_name, type(self).__name__)
        pooled = cast(Tensor, self.model(require_tensor(image, name=self.input_name)))
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


def _classifier_prefixes(model: nn.Module) -> tuple[str, ...]:
    """The keys this architecture calls its classifier, asked of the architecture rather than listed here.

    Every timm model publishes the attribute its own classifier sits at, and it survives being built
    headless (measured: ``num_classes=0`` keeps ``pretrained_cfg['classifier'] == 'fc'`` on a resnet).
    A table of families here would be a second statement of something timm already says, and it would
    fall behind the first the week timm adds a family.
    """
    named = cast("dict[str, Any]", getattr(model, "pretrained_cfg", {})).get("classifier", "fc")
    names = named if isinstance(named, tuple | list) else (named,)
    return tuple(f"{name}." for name in names)
