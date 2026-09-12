"""The names an `export` declaration may write."""

from __future__ import annotations

from src.core import Registry
from src.export.base import Exporter

exporter_registry: Registry[Exporter] = Registry("exporter")
"""What `export` writes: the formats a trained model is shipped in.

Each backend registers beside its own definition, and a format this framework never heard of is one
``_target_`` away — it only has to write a file and read it back.
"""
