"""The loggers capability: experiment trackers behind Lightning's ``Logger``."""

from __future__ import annotations

from src.loggers.clearml import ClearMLLogger
from src.loggers.registry import logger_registry

__all__ = ["ClearMLLogger", "logger_registry"]
