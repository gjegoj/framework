"""Composition helpers: pure wiring functions used by the root main.py."""

from src.composition.wiring import (
    build_backbone,
    build_bindings,
    build_data_module,
    build_data_source,
    build_optimizer_builder,
    build_staged_sources,
    build_tasks,
    build_transforms,
)

__all__ = [
    "build_backbone",
    "build_bindings",
    "build_data_module",
    "build_data_source",
    "build_optimizer_builder",
    "build_staged_sources",
    "build_tasks",
    "build_transforms",
]
