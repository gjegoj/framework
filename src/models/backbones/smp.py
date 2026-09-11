"""smp as a backbone: an encoder-decoder pair as two streams, with its segmentation head on offer."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any, cast

import segmentation_models_pytorch as smp
from torch import Tensor, nn

from src.core import Axis, Modality, Stream, TensorShape, TensorTree, require_tensor
from src.models.base import Backbone, reads
from src.models.registry import backbone_registry

log = logging.getLogger(__name__)

PREFIX_TOKEN_HOOK = "_forward_with_prefix_tokens"
"""The smp hook that reads a ViT's intermediate features; replaced to add the encoder's final norm."""


@backbone_registry.register("smp")
class SmpBackbone(Backbone):
    """Publishes ``encoder`` (the last encoder stage) and ``decoder`` (the full-size map).

    smp's own segmentation head leaves the forward path so that this run's heads apply instead; it is
    kept as an unregistered template ``native_head`` rebuilds at any number of classes. That is what
    lets one backbone serve segmentation from the decoder and classification from the encoder at once.

    Args:
        arch: smp architecture, e.g. ``"unet"``, ``"dpt"``.
        encoder_name: Encoder backbone, e.g. ``"resnet18"``, or a timm ViT for DPT.
        pretrained: Load the encoder's ImageNet weights; ``encoder_weights`` overrides this.
        input_name: Which of the batch's inputs to encode.
        **options: Forwarded to ``smp.create_model``.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        input_name: str = Modality.IMAGE,
        **options: Any,
    ) -> None:
        super().__init__()
        options.setdefault("encoder_weights", "imagenet" if pretrained else None)
        # One class is a placeholder: the head is taken off the graph right below.
        model = smp.create_model(arch=arch, encoder_name=encoder_name, classes=1, **options)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.input_name = input_name
        # A one-tuple hides the template from module registration: its weights never run in a forward
        # pass, so they must not appear in a checkpoint, a summary or an export.
        self._template: tuple[nn.Module] = (model.segmentation_head,)
        self.encoder_width = int(model.encoder.out_channels[-1])
        # Every smp head starts with a Conv2d over the decoder's output, whatever the architecture.
        self.decoder_width = _first_conv(model.segmentation_head).in_channels
        # DPT-style encoders return their spatial features beside the prefix tokens their decoder reads;
        # asking the encoder is safer than matching an architecture name against a list.
        self.carries_prefix_tokens = bool(getattr(model.encoder, "has_prefix_tokens", False))
        if self.carries_prefix_tokens:
            _apply_the_encoders_final_norm(self.encoder, encoder_name)

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {
            Stream.ENCODER: _map(self.encoder_width),
            Stream.DECODER: _map(self.decoder_width),
        }

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        pictures = require_tensor(reads(inputs, self.input_name, type(self).__name__), name=self.input_name)
        encoded = self.encoder(pictures)
        if self.carries_prefix_tokens:
            stages, prefix_tokens = encoded
            return {Stream.ENCODER: stages[-1], Stream.DECODER: self.decoder(stages, prefix_tokens)}
        return {Stream.ENCODER: encoded[-1], Stream.DECODER: self.decoder(encoded)}

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        """smp's segmentation head over the decoder, or its pooling classifier over the encoder."""
        if stream == Stream.DECODER:
            head = copy.deepcopy(self._template[0])
            _resize_last_projection(head, out_features)
            return head
        if stream == Stream.ENCODER:
            from segmentation_models_pytorch.base import ClassificationHead

            return cast(nn.Module, ClassificationHead(self.encoder_width, classes=out_features, pooling="avg"))
        return None


def _apply_the_encoders_final_norm(encoder: nn.Module, encoder_name: str) -> None:
    """Read a ViT's intermediate features through its own final LayerNorm, as the DINO recipe does.

    smp calls ``forward_intermediates`` without ``norm`` and exposes no flag for it, so its hook is
    replaced. Measured on smp 0.5.0 with a ViT-S: the last stage goes from std 0.84 to 1.0, which is
    what a DPT decoder trained on normalized features expects. The way in is private, so a version
    that moves it leaves a warning rather than a run that quietly reassembles unnormalized features.
    """
    inner, indices = getattr(encoder, "model", None), getattr(encoder, "_output_indices", None)
    if inner is None or indices is None or not hasattr(encoder, PREFIX_TOKEN_HOOK):
        log.warning(
            "Encoder %r carries prefix tokens, but this version of smp does not expose %s over a timm model: "
            "its intermediate features reach the decoder without the encoder's final norm.",
            encoder_name,
            PREFIX_TOKEN_HOOK,
        )
        return

    def read_with_norm(pictures: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        stages = inner.forward_intermediates(
            pictures, indices=indices, intermediates_only=True, return_prefix_tokens=True, norm=True
        )
        return [stage for stage, _ in stages], [tokens for _, tokens in stages]

    setattr(encoder, PREFIX_TOKEN_HOOK, read_with_norm)


def _map(width: int) -> TensorShape:
    """A feature map: its width is known at build, its extent only once a picture arrives."""
    return TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(width, None, None))


def _first_conv(module: nn.Module) -> nn.Conv2d:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError(f"No Conv2d in {type(module).__name__}; the decoder's width cannot be read from its head.")


def _resize_last_projection(head: nn.Module, out_features: int) -> None:
    """Swap the head's last projection for one of the same kind with ``out_features`` outputs."""
    parents = [(parent, name, child) for parent in head.modules() for name, child in parent.named_children()]
    projections = [(parent, name, child) for parent, name, child in parents if isinstance(child, nn.Conv2d | nn.Linear)]
    if not projections:
        raise ValueError(f"No projection in {type(head).__name__}; it cannot be rebuilt at another size.")
    parent, name, layer = projections[-1]
    setattr(parent, name, _resized(layer, out_features))


def _resized(layer: nn.Conv2d | nn.Linear, out_features: int) -> nn.Module:
    if isinstance(layer, nn.Linear):
        return nn.Linear(layer.in_features, out_features, bias=layer.bias is not None)
    return nn.Conv2d(
        layer.in_channels,
        out_features,
        kernel_size=cast(Any, layer.kernel_size),
        stride=cast(Any, layer.stride),
        padding=cast(Any, layer.padding),
        bias=layer.bias is not None,
    )
