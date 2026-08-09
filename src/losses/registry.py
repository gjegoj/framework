"""Registries of the losses capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import Criterion

criterion_registry: Registry[Criterion] = Registry("criterion")
"""Config-facing loss criteria; register with ``@criterion_registry.register("name")``."""
