"""Data sources: adapters that read annotation files into a DataFrame.

A ``DataSource`` is the only place that knows a concrete file format. Splitting,
sampling and per-row decoding all operate on the returned DataFrame, so adding a
new format (JSON, parquet, ...) is a localized change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.core.registry import Registry

data_sources: Registry[DataSource] = Registry("data_source")


class DataSource(ABC):
    """Reads one or more annotation files into a single DataFrame."""

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Return all annotations as one concatenated DataFrame."""


@data_sources.register("csv")
class CsvDataSource(DataSource):
    """Reads and concatenates one or more CSV files.

    Parameters:
        paths (str | list[str]): One path or a list of CSV paths to concatenate.
    """

    def __init__(self, paths: str | list[str]) -> None:
        self._paths = [paths] if isinstance(paths, str) else list(paths)
        if not self._paths:
            raise ValueError("CsvDataSource requires at least one path.")

    def read(self) -> pd.DataFrame:
        frames = []
        for path in self._paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"CSV source not found: {path}")
            frames.append(pd.read_csv(path))
        return pd.concat(frames, ignore_index=True)
