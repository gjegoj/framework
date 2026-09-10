"""Data preparation interfaces; source adapters remain optional."""

from __future__ import annotations

from src.data.datamodules.base import DataModule
from src.data.encoders.base import TargetEncoder
from src.data.preprocessing import Preprocessor, Stateful

__all__ = ["DataModule", "Preprocessor", "Stateful", "TargetEncoder"]
