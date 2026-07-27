"""Model-layer wiring: the backbone adapter and LoRA application."""

from __future__ import annotations

from src.composition.wiring.common import forward_extras
from src.config.schema import BackboneConfig, ExperimentConfig
from src.core.ports import Backbone
from src.models.assembly import CompositeModel
from src.models.backbones import MultiEncoderBackbone
from src.models.lora import apply_lora
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


def apply_lora_if_configured(config: ExperimentConfig, model: CompositeModel) -> None:
    """Inject LoRA adapters into the model's backbone when ``config.lora`` is set.

    Heads are never wrapped — they stay fully trainable; the facade freezes the
    backbone's base weights and leaves only ``lora_*`` parameters trainable.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        model (CompositeModel): Assembled student model (mutated in place).
    """
    if config.lora is None:
        return
    apply_lora(model.backbone, config.lora)


def validate_lora_preconditions(config: ExperimentConfig) -> None:
    """Reject configs where the freeze callback would kill LoRA training.

    ``BaseFinetuning.freeze`` sets ``requires_grad=False`` recursively, so a freeze
    target inside the LoRA-injected backbone would silently freeze the adapters too.
    Rule: LoRA owns backbone freezing.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Raises:
        ValueError: If ``lora:`` is set and a ``freeze`` callback targets the backbone.
    """
    if config.lora is None or not config.callbacks:
        return
    freeze_options = config.callbacks.get("freeze") or {}
    targets: list[str] = list(freeze_options.get("targets", []))
    overlapping = [target for target in targets if target.startswith("model.backbone")]
    if overlapping:
        raise ValueError(
            f"LoRA owns backbone freezing: freeze targets {overlapping} point inside model.backbone, "
            "which would freeze the LoRA adapters too. Remove the freeze callback or retarget it."
        )
