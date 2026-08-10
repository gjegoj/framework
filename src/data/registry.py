"""Registries of the data capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import DataModule
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

vendor_data_module_registry: Registry[DataModule] = Registry("vendor data module")
"""The pipeline a vendor family reads with, under the same key that family's model has.

A family that arrives whole brings both halves — its network *and* the loader whose
augmentation is box-aware — and a run names it once, in ``config.model``. Assembly then
needs to reach the pipeline from that one name; before this it reached a class it had
been told about in the composition root, so a second family meant editing the root.

Keyed to match ``vendor_model_registry`` rather than derived from it, because the two live in
capability packages that do not import one another. A key present in one and missing from
the other fails at assembly, naming what *is* registered — and a test pins the pairing so
the mismatch is caught before a run ever asks.
"""
