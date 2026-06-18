"""Model-layer wiring: the backbone adapter."""

from __future__ import annotations

from src.composition.wiring.common import forward_extras
from src.config.schema import BackboneConfig
from src.core.ports import Backbone
from src.models.backbones import MultiEncoderBackbone
from src.models.registry import backbones

_BACKBONE_CORE_FIELDS = frozenset({"kind", "name", "pretrained"})


def build_backbone(backbone_config: BackboneConfig) -> Backbone:
    """Build the backbone from config, forwarding adapter-specific extras.

    ``kind`` selects the adapter; ``name``/``pretrained`` are passed explicitly
    and any extra fields (e.g. smp's ``encoder_name``) are forwarded as keyword
    args.  ``kind: multi`` is a composite — its sub-encoders are built recursively
    and wrapped in a ``MultiEncoderBackbone``.

    Parameters:
        backbone_config (BackboneConfig): Validated backbone config (extras allowed).

    Returns:
        Backbone: The constructed backbone adapter.
    """
    if backbone_config.kind == "multi":
        return _build_multi_encoder(backbone_config)
    extra = forward_extras(backbone_config, _BACKBONE_CORE_FIELDS)
    return backbones.create(
        backbone_config.kind, name=backbone_config.name, pretrained=backbone_config.pretrained, **extra
    )


def _build_multi_encoder(backbone_config: BackboneConfig) -> MultiEncoderBackbone:
    """Build a multi-encoder backbone, constructing each sub-encoder recursively.

    The ``encoders`` field is a mapping ``{name: backbone-spec}``; each spec is
    re-validated as a ``BackboneConfig`` and built through ``build_backbone``, so
    any backbone kind (timm/smp/embedding/...) can serve as a sub-encoder.
    """
    raw = backbone_config.model_dump()
    encoders = {name: build_backbone(BackboneConfig(**spec)) for name, spec in raw["encoders"].items()}
    return MultiEncoderBackbone(encoders=encoders, embed_dim=raw.get("embed_dim"))
