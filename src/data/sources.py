"""Data sources: adapters that read annotation files into a DataFrame.

A ``DataSource`` is the only place that knows a concrete file format. Splitting,
sampling and per-row decoding all operate on the returned DataFrame, so adding a
new format is a localized change: subclass ``FileDataSource`` and implement
``_read_file``. The multi-path normalisation, existence check and concatenation
live once in the base (Template Method).
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


class FileDataSource(DataSource):
    """Base for file-backed sources: normalise paths, check, read, concatenate.

    Subclasses implement ``_read_file`` for their format; the base handles
    accepting one path or many, validating existence, and concatenating frames.

    Parameters:
        paths (str | list[str]): One path or a list of paths to concatenate.
    """

    def __init__(self, paths: str | list[str]) -> None:
        self._paths = [paths] if isinstance(paths, str) else list(paths)
        if not self._paths:
            raise ValueError(f"{type(self).__name__} requires at least one path.")

    @abstractmethod
    def _read_file(self, path: str) -> pd.DataFrame:
        """Read a single file of this source's format into a DataFrame."""

    def read(self) -> pd.DataFrame:
        frames = []
        for path in self._paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Data source file not found: {path}")
            frames.append(self._read_file(path))
        return pd.concat(frames, ignore_index=True)


@data_sources.register("csv")
class CsvDataSource(FileDataSource):
    """Reads and concatenates one or more CSV files."""

    def _read_file(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)


@data_sources.register("json")
class JsonDataSource(FileDataSource):
    """Reads and concatenates one or more JSON files (array of record objects)."""

    def _read_file(self, path: str) -> pd.DataFrame:
        return pd.read_json(path)
