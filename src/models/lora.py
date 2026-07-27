"""LoRA facade over peft: in-place injection, freezing, merge for export.

peft is a detail of this module only. Injection uses ``inject_adapter_in_model``
(no ``PeftModel`` wrapper), so the surrounding model keeps its class and natural
``state_dict`` keys; targeted layers gain ``lora_A``/``lora_B`` children.
"""

from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig
from peft import inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from torch import nn

from src.config.schema import LoraConfig


def apply_lora(backbone: nn.Module, lora_config: LoraConfig) -> None:
    """Inject LoRA adapters into ``backbone`` in place and freeze everything else.

    Parameters:
        backbone (nn.Module): Backbone whose matching Linear/Conv2d layers get adapters.
        lora_config (LoraConfig): Typed section; extras forward to peft verbatim.

    Raises:
        ValueError: If ``target_modules`` matched no module (a typo must not silently
            train the full backbone).
    """
    peft_config = PeftLoraConfig(
        r=lora_config.rank,
        # peft annotates lora_alpha as int but only ever divides by r at runtime — keep float freedom.
        lora_alpha=lora_config.alpha,  # type: ignore[arg-type]
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        **(lora_config.model_extra or {}),
    )
    inject_adapter_in_model(peft_config, backbone)
    if not has_lora_layers(backbone):
        raise ValueError(
            f"LoRA: no module matched target_modules={lora_config.target_modules}; "
            "check the names against the backbone's named_modules()."
        )
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad = "lora_" in name


def merge_lora(model: nn.Module) -> None:
    """Fold every adapter into its base layer and remove peft modules from the tree.

    After this the model contains only plain layers — required before export tracing
    so the deployed graph carries no LoRA overhead.

    Parameters:
        model (nn.Module): Model (or backbone) whose adapters are merged in place.
    """
    lora_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    for name, module in lora_layers:
        module.merge(safe_merge=True)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.get_base_layer())


def has_lora_layers(model: nn.Module) -> bool:
    """Whether any peft LoRA layer is present in ``model``'s tree.

    Parameters:
        model (nn.Module): Model to scan.
    """
    return any(isinstance(module, LoraLayer) for module in model.modules())
