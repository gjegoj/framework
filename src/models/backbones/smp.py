"""smp backbone adapter: an encoder+decoder pair as a two-stream feature source."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, override

import segmentation_models_pytorch as smp
from torch import nn

from src.core.entities import Features
from src.core.ports import Backbone
from src.core.taxonomy import Modality, Stream
from src.models.backbones.checkpoints import (
    SMP_HEAD_PREFIXES,
    load_arrived_weights,
    replace_last_projection,
    transplanted_segmentation_head,
)
from src.models.registry import backbone_registry

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import Tensor

log = logging.getLogger(__name__)


@backbone_registry.register("smp")
class SmpBackbone(Backbone):
    """Encoder+decoder from smp, exposing ``Stream.ENCODER`` and ``Stream.DECODER``.

    smp's built-in segmentation head leaves the forward path, so the framework's per-task
    heads apply; the original head is kept as an unregistered template ``native_head`` can
    rebuild at any ``out_features``. ``ENCODER`` is the last encoder stage ``[B, D, H', W']``,
    ``DECODER`` the full decoder output ``[B, D, H, W]``. For DPT with ViT encoders, the
    encoder's final LayerNorm is applied to the intermediate features (upstream skips it).

    Parameters:
        arch (str): smp architecture, e.g. ``"unet"``, ``"dpt"``.
        encoder_name (str): Encoder backbone, e.g. ``"resnet34"``, or a timm ViT for DPT.
        pretrained (bool): Load ImageNet encoder weights; moot when ``checkpoint_path`` is given.
        input_name (str): Which batch input to encode.
        checkpoint_path (str | Path | None): A full smp model's checkpoint; encoder and
            decoder load, the segmentation head is stashed for ``native_head``.
        use_ema (bool): Prefer the checkpoint's EMA branch when present.
        strict (bool): Refuse a checkpoint that does not match exactly; ``False`` allows partial loads.
        weights_only (bool): ``torch.load`` safety knob, forwarded.
        **kwargs: Forwarded verbatim to ``smp.create_model``.
    """

    def __init__(
        self,
        arch: str = "unet",
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        input_name: str = Modality.IMAGE,
        checkpoint_path: str | Path | None = None,
        use_ema: bool = True,
        strict: bool = True,
        weights_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._architecture = f"{arch}-{encoder_name}"
        hub_default = "imagenet" if pretrained and checkpoint_path is None else None
        encoder_weights = kwargs.pop("encoder_weights", hub_default)
        model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=1,  # Placeholder: the head is dropped; only encoder+decoder are kept.
            **kwargs,
        )
        # The placeholder head leaves the model before any loading: its keys are
        # stashed by the filter, so a strict load must not expect them either.
        # The original module survives as the cloning template for native_head.
        head_template = model.segmentation_head
        model.segmentation_head = nn.Identity()
        if getattr(model, "classification_head", None) is not None:
            model.classification_head = nn.Identity()
        self._carried_head: dict[str, Tensor] | None = None
        if checkpoint_path is not None:
            self._carried_head = load_arrived_weights(
                model,
                f"{arch}/{encoder_name}",
                checkpoint_path,
                use_ema=use_ema,
                strict=strict,
                weights_only=weights_only,
                stash_prefixes=SMP_HEAD_PREFIXES,
            )
        self._input_name = input_name
        self._encoder = model.encoder
        self._decoder = model.decoder
        # Kept ONLY as a cloning template for native_head. The 1-tuple hides it
        # from module registration: its weights never run in forward, so they
        # must not appear in checkpoints, summaries, or exports.
        self._head_template: tuple[nn.Module] = (head_template,)
        self._encoder_dim = int(model.encoder.out_channels[-1])
        # Every smp head starts with Conv2d(decoder_out_channels, ...), so the
        # first Conv2d's in_channels is the decoder width — for any head type.
        self._decoder_dim = _first_conv_in_channels(head_template)
        # DPT is the only smp arch whose encoder returns (features, prefix_tokens)
        # and whose decoder takes both.
        self._dpt_style = arch.lower() == "dpt"
        if self._dpt_style:
            self._normalize_prefix_token_intermediates(encoder_name)

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        encoder_output = self._encoder(inputs[self._input_name])
        if self._dpt_style:
            spatial_features, prefix_tokens = encoder_output[0], encoder_output[1]
            decoder_output = self._decoder(spatial_features, prefix_tokens)
            encoder_last = spatial_features[-1]
        else:
            decoder_output = self._decoder(encoder_output)
            encoder_last = encoder_output[-1]
        return Features(streams={Stream.ENCODER: encoder_last, Stream.DECODER: decoder_output})

    @property
    @override
    def architecture(self) -> str:
        """Composed rather than asked: measured, smp calls a Unet on a ResNet-34 `u-resnet34`."""
        return self._architecture

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.ENCODER: self._encoder_dim, Stream.DECODER: self._decoder_dim}

    @override
    def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
        if stream == Stream.DECODER:
            if self._carried_head is not None:
                return transplanted_segmentation_head(self._head_template[0], self._carried_head, out_features)
            head = copy.deepcopy(self._head_template[0])
            replace_last_projection(head, out_features)
            return head
        if stream == Stream.ENCODER:
            from segmentation_models_pytorch.base import ClassificationHead

            classifier: nn.Module = ClassificationHead(in_channels=in_features, classes=out_features, pooling="avg")
            return classifier
        return None

    def _normalize_prefix_token_intermediates(self, encoder_name: str) -> None:
        """Apply the ViT's final LayerNorm to intermediate features (DINO recipe).

        smp calls ``forward_intermediates`` without ``norm`` and exposes no
        flag for it, so the encoder's private hook is wrapped. A no-op for
        encoders without prefix tokens (CNNs).
        """
        encoder: Any = self._encoder
        if not getattr(encoder, "has_prefix_tokens", False):
            return
        log.info(
            "Encoder '%s' carries prefix tokens; applying its final LayerNorm to "
            "intermediate features before DPT reassembly (DINO recipe).",
            encoder_name,
        )
        model, indices = encoder.model, encoder._output_indices

        def forward_with_norm(x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
            outputs = model.forward_intermediates(
                x, indices=indices, intermediates_only=True, return_prefix_tokens=True, norm=True
            )
            return [output[0] for output in outputs], [output[1] for output in outputs]

        encoder._forward_with_prefix_tokens = forward_with_norm


def _first_conv_in_channels(module: nn.Module) -> int:
    """Return ``in_channels`` of the first ``Conv2d`` in depth-first order."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            return int(layer.in_channels)
    raise ValueError(f"No Conv2d found in {type(module).__name__}; cannot infer the decoder width.")
