"""The concrete ``DataModule`` implementations, one module per kind of layout."""

from __future__ import annotations

from src.data.datamodules.base import DataModule, require_stage
from src.data.datamodules.table import DeclaredSource, StageDataset, TableDataModule

__all__ = [
    "DataModule",
    "DeclaredSource",
    "StageDataset",
    "TableDataModule",
    "require_stage",
]
