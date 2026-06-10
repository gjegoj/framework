"""Model-layer wiring: the backbone adapter."""

from __future__ import annotations

from src.composition.wiring.common import forward_extras
from src.config.schema import BackboneConfig
from src.core.ports import Backbone
from src.models.registry import backbones

_BACKBONE_CORE_FIELDS = frozenset({"kind", "name", "pretrained"})


def build_backbone(backbone_cfg: BackboneConfig) -> Backbone:
    """Build the backbone from config, forwarding adapter-specific extras.

    ``kind`` selects the registry adapter; ``name``/``pretrained`` are passed
    explicitly and any extra fields (e.g. smp's ``encoder_name``) are forwarded
    as keyword args.

    Parameters:
        backbone_cfg (BackboneConfig): Validated backbone config (extras allowed).

    Returns:
        Backbone: The constructed backbone adapter.
    """
    extra = forward_extras(backbone_cfg, _BACKBONE_CORE_FIELDS)
    return backbones.create(backbone_cfg.kind, name=backbone_cfg.name, pretrained=backbone_cfg.pretrained, **extra)
