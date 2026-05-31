"""Registries for pluggable model components (extension points for users)."""

from src.core.ports import Backbone, Head
from src.core.registry import Registry

backbones: Registry[Backbone] = Registry("backbone")
head_builders: Registry[Head] = Registry("head")
