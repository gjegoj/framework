"""The names a `tracker` declaration may write."""

from __future__ import annotations

from lightning.pytorch.loggers import Logger

from src.core import Registry

tracker_registry: Registry[Logger] = Registry("tracker")
"""What `tracker` writes: where a run's numbers are recorded.

Each backend registers beside its own definition, and anything else is one ``_target_`` away.
"""
