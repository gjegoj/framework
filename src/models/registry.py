"""Registries of the models capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import Backbone, Head, Model
    from src.models.adapters import Adapters

head_registry: Registry[Head] = Registry("head")
"""Config-facing heads; register with ``@head_registry.register("name")``."""

backbone_registry: Registry[Backbone] = Registry("backbone")
"""Config-facing backbones; register with ``@backbone_registry.register("name")``."""

model_registry: Registry[Model] = Registry("model")
"""Config-facing model families that arrive whole — head, loss and decoding their own.

Separate from ``backbone_registry`` because the two answer different questions: a
backbone is a piece this framework composes with, a model is a family it delegates to.
A name found here rather than there is what tells assembly which of the two it is
looking at, and is the only place that decision is made.
"""

adapter_registry: Registry[Adapters] = Registry("adapters")
"""Config-facing parameter-efficient techniques; register with ``@adapter_registry.register("name")``."""
