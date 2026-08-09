"""Experiment-tracker backends by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

logger_registry: Registry[Logger] = Registry("logger")
"""Lightning ``Logger`` subclasses by config-facing name.

Assembly builds the declared one through ``instantiate``, so every constructor knob is
reachable from config. An adapter imports its third-party client lazily: a registered
backend must not require its package until it is actually built.
"""
