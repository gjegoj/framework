"""SmpBackbone: a segmentation-models-pytorch encoder+decoder as a feature source.

We keep smp's encoder and decoder but drop its built-in segmentation head, so the
backbone produces a dense ``decoder`` feature map ``[B, D, H, W]`` that our own
``ConvHead`` turns into class logits — preserving the framework's
backbone→FeatureBundle→head split (and per-head LR / multitask on one backbone).

## Streams

| Key              | Shape            | Description                                             |
|------------------|------------------|---------------------------------------------------------|
| ``encoder_last`` | ``[B, D, H, W]`` | Last encoder stage features (no pooling). Use with      |
|                  |                  | ``prefer_native=True`` to get smp's ClassificationHead  |
|                  |                  | (has adaptive-avg-pool inside), or supply your own head.|
| ``decoder``      | ``[B, D, H, W]`` | Full decoder output ready for a segmentation head.      |

**Typical multitask config (smp backbone, segmentation + classification):**

.. code-block:: yaml

    backbone:
      kind: smp
      name: unet
      encoder_name: resnet34

    tasks:
      mask:
        preset: segmentation
        target: mask_path
        class_mapping: {0: background, 1: defect}
        # feature_key: decoder  ← default for DenseTopology

      label:
        preset: classification
        target: label
        class_mapping: {0: cat, 1: dog}
        feature_key: encoder_last   # ← uses smp's ClassificationHead (pools internally)

The original smp segmentation head is retained as a template so ``native_head``
can reconstruct it with the correct ``out_channels``.  This works generically for
any smp architecture (``SegmentationHead``, ``DPTSegmentationHead``, etc.) by
finding the last ``Conv2d``/``Linear`` in the head's module tree and replacing
its output dimension — no ``isinstance`` checks on the head type required.
"""

from __future__ import annotations

import copy
from typing import Any

import segmentation_models_pytorch as smp
from torch import Tensor, nn

from src.core.entities import FeatureBundle
from src.core.keys import DECODER, ENCODER_LAST, IMAGE
from src.core.ports import Backbone
from src.models.registry import backbones

_AVAILABLE_KEYS = (ENCODER_LAST, DECODER)


@backbones.register("smp")
class SmpBackbone(Backbone):
    """Encoder+decoder from smp, exposing ``encoder_last`` and ``decoder`` streams.

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
            classes=1,  # placeholder: we drop smp's head and keep only encoder+decoder
            **kwargs,
        )
        self._encoder = model.encoder
        self._decoder = model.decoder
        self._seg_head_template: nn.Module = model.segmentation_head
        self._encoder_last_dim = int(model.encoder.out_channels[-1])

        # DPT is the only smp architecture whose encoder returns
        # (features_list, prefix_tokens_list) and whose decoder signature is
        # decoder(features, prefix_tokens) instead of decoder(features).
        self._dpt_style = name.lower() == "dpt"

        # The segmentation head always starts with Conv2d(decoder_out_channels, …),
        # so its first Conv2d's in_channels == the decoder's output channels.
        # This works for SegmentationHead, DPTSegmentationHead, and any future
        # head type — no isinstance checks, no dummy forward pass.
        self._decoder_dim = _first_conv_in_channels(model.segmentation_head)

    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        encoder_output = self._encoder(inputs[IMAGE])
        if self._dpt_style:
            # DPT encoder returns (spatial_features, prefix_tokens) where each is
            # a list of N tensors — one per decoder reassembly stage.
            spatial_features, prefix_tokens = encoder_output[0], encoder_output[1]
            decoder_output = self._decoder(spatial_features, prefix_tokens)
            encoder_last = spatial_features[-1]  # [B, D, H/patch, W/patch]
        else:
            decoder_output = self._decoder(encoder_output)
            encoder_last = encoder_output[-1]  # [B, D, H, W]
        return FeatureBundle({ENCODER_LAST: encoder_last, DECODER: decoder_output})

    def feature_dim(self, key: str) -> int:
        if key == ENCODER_LAST:
            return self._encoder_last_dim
        if key == DECODER:
            return self._decoder_dim
        available_keys = ", ".join(f"'{k}'" for k in _AVAILABLE_KEYS)
        raise KeyError(
            f"SmpBackbone exposes: {available_keys}. Got: {key!r}. Check the backbone docstring for stream shapes."
        )

    def native_head(self, feature_key: str, in_features: int, out_features: int) -> nn.Module | None:
        """Return a backbone-native head for the given stream.

        - ``decoder`` → deep copy of smp's segmentation head with ``out_channels`` replaced.
          Works for any smp head architecture (SegmentationHead, DPTSegmentationHead, …).
        - ``encoder_last`` → ``smp.base.ClassificationHead`` with adaptive-avg-pool inside.
        - Anything else → ``None`` (use the head registry instead).
        """
        if feature_key == DECODER:
            segmentation_head = copy.deepcopy(self._seg_head_template)
            _replace_last_projection(segmentation_head, out_features)
            return segmentation_head
        if feature_key == ENCODER_LAST:
            from segmentation_models_pytorch.base import ClassificationHead

            classification_head: nn.Module = ClassificationHead(
                in_channels=in_features,
                classes=out_features,
                pooling="avg",
            )
            return classification_head
        return None


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _first_conv_in_channels(module: nn.Module) -> int:
    """Return ``in_channels`` of the first ``Conv2d`` found by depth-first traversal."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            return int(layer.in_channels)
    raise ValueError(f"No Conv2d found in {type(module).__name__}; cannot infer decoder output channels.")


def _replace_last_projection(module: nn.Module, out_channels: int) -> None:
    """Replace the last Conv2d or Linear output dimension in ``module`` in-place."""
    last = _find_last_projection(module)
    if last is None:
        raise ValueError(f"No Conv2d or Linear found in {type(module).__name__}.")
    parent, attr_name, layer = last
    setattr(parent, attr_name, _clone_with_new_out_channels(layer, out_channels))


def _find_last_projection(
    module: nn.Module,
) -> tuple[nn.Module, str, nn.Conv2d | nn.Linear] | None:
    """Return (parent, attr_name, layer) for the last Conv2d or Linear in ``module``."""
    found: list[tuple[nn.Module, str, nn.Conv2d | nn.Linear]] = []

    def _visit(mod: nn.Module) -> None:
        for child_name, child in mod.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                found.append((mod, child_name, child))
            else:
                _visit(child)

    _visit(module)
    return found[-1] if found else None


def _clone_with_new_out_channels(
    layer: nn.Conv2d | nn.Linear,
    out_channels: int,
) -> nn.Module:
    """Return a new layer identical to ``layer`` except with ``out_channels`` outputs."""
    if isinstance(layer, nn.Conv2d):
        return nn.Conv2d(
            layer.in_channels,
            out_channels,
            kernel_size=layer.kernel_size,  # type: ignore[arg-type]
            stride=layer.stride,  # type: ignore[arg-type]
            padding=layer.padding,  # type: ignore[arg-type]
            bias=layer.bias is not None,
        )
    return nn.Linear(
        layer.in_features,
        out_channels,
        bias=layer.bias is not None,
    )
