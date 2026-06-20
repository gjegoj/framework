"""Registries for pluggable model components (extension points for users)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import Backbone, Head

backbones: Registry[Backbone] = Registry("backbone")
head_builders: Registry[Head] = Registry("head")
