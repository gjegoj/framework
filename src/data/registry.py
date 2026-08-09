"""Registries of the data capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.data.cache import LoaderCache
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader
    from src.data.sources import TableSource

table_source_registry: Registry[TableSource] = Registry("table source")
"""Config-facing table sources; register with ``@table_source_registry.register("name")``."""

input_loader_registry: Registry[InputLoader] = Registry("input loader")
"""Config-facing input loaders; register with ``@input_loader_registry.register("name")``."""

target_encoder_registry: Registry[TargetEncoder] = Registry("target encoder")
"""Config-facing target encoders; register with ``@target_encoder_registry.register("name")``."""

cache_registry: Registry[LoaderCache] = Registry("cache")
"""Config-facing caches; register with ``@cache_registry.register("name")``."""
