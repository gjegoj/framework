"""Registries of the models capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from torch import nn

    from src.core.ports import Backbone
    from src.models.adapters import Adapters

head_registry: Registry[nn.Module] = Registry("head")
"""Config-facing heads — any ``nn.Module`` built at ``(in_features, out_features)``; register with ``@head_registry.register("name")``."""

backbone_registry: Registry[Backbone] = Registry("backbone")
"""Config-facing backbones; register with ``@backbone_registry.register("name")``."""

adapter_registry: Registry[Adapters] = Registry("adapter")
"""Config-facing parameter-efficient techniques; register with ``@adapter_registry.register("name")``."""
