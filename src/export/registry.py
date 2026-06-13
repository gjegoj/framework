"""Exporter registry — maps format name to ``ModelExporter`` implementation."""

from __future__ import annotations

from src.core.ports import ModelExporter
from src.core.registry import Registry

exporters: Registry[ModelExporter] = Registry("exporter")
