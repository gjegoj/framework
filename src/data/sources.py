"""File-backed tables in the formats a suffix names, plus a reproducible cap for smoke runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.data.base import Table, TableSource
from src.data.registry import table_source_registry

SUFFIX_FORMATS: Final = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}
CAP_SEED: Final = 42


def format_of(path: str | Path) -> str:
    """The registered format a file suffix implies; anything else must declare its format."""
    suffix = Path(path).suffix.lower()
    try:
        return SUFFIX_FORMATS[suffix]
    except KeyError:
        known = ", ".join(sorted(table_source_registry))
        raise LookupError(f"Cannot infer the table format of {str(path)!r}; declare one of: {known}.") from None


def source_for(paths: str | Path | Sequence[str | Path], *, format: str | None = None, **reader: Any) -> TableSource:
    """One source over one or several files of the same format."""
    listed = [paths] if isinstance(paths, str | Path) else list(paths)
    if not listed:
        raise ValueError("A source needs at least one path.")
    factory: Callable[..., TableSource] = table_source_registry.get(format or format_of(listed[0]))
    return factory(listed, **reader)


class FileSource(TableSource):
    """Several files of one format, concatenated in the declared order; reader keywords forward to pandas."""

    def __init__(self, paths: Sequence[str | Path], **reader: Any) -> None:
        self.paths = [Path(path) for path in paths]
        self.reader = reader

    def read(self) -> Table:
        frames = [self._read_file(path) for path in self.paths]
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    def _read_file(self, path: Path) -> Table:
        raise NotImplementedError


@table_source_registry.register("csv")
class CsvSource(FileSource):
    def _read_file(self, path: Path) -> Table:
        return pd.read_csv(path, **self.reader)


@table_source_registry.register("json")
class JsonSource(FileSource):
    def _read_file(self, path: Path) -> Table:
        return pd.read_json(path, **self.reader)


@table_source_registry.register("jsonl")
class JsonLinesSource(FileSource):
    def _read_file(self, path: Path) -> Table:
        return pd.read_json(path, lines=True, **self.reader)


def capped(table: Table, max_samples: int | float | None) -> Table:
    """A random, seeded subset: a count, or a fraction of the rows; None keeps everything."""
    if max_samples is None:
        return table
    if max_samples <= 0 or (isinstance(max_samples, float) and max_samples > 1.0):
        raise ValueError(f"max_samples is a positive count or a fraction up to 1.0, got {max_samples}.")
    if isinstance(max_samples, float):
        kept = table.sample(frac=max_samples, random_state=CAP_SEED)
    else:
        kept = table.sample(n=min(max_samples, len(table)), random_state=CAP_SEED)
    return kept.reset_index(drop=True)
