"""Exporter registry — maps format name to ``ModelExporter`` implementation."""

from __future__ import annotations

from src.core.registry import Registry
from src.export.ports import ModelExporter

exporters: Registry[ModelExporter] = Registry("exporter")
