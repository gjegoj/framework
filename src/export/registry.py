"""Deployment formats by name — the export capability's single extension point."""

from __future__ import annotations

from src.core import Registry
from src.export.exporters import Exporter

exporter_registry: Registry[Exporter] = Registry("exporter")
