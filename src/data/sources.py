"""File-backed tables in the formats a suffix names, plus a reproducible cap for smoke runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from src.data.base import Table, TableSource
from src.data.registry import table_source_registry

CAP_SEED = 42
"""Which rows a cap keeps, fixed and separate from the experiment's seed — for the same reason
``Split.seed`` is: two runs at different seeds must read the same slice, or their numbers are not
comparable.
"""


def format_of(path: str | Path) -> str:
    """The registered format a file suffix implies; anything else must declare its format.

    Read off the registry rather than a table beside it: a source that says which suffixes it reads
    cannot be registered and left unreachable, and the names this refusal lists are the ones that work.
    """
    suffix = Path(path).suffix.lower()
    for name in table_source_registry:
        source = table_source_registry.get(name)
        if issubclass(source, FileSource) and suffix in source.suffixes:
            return name
    known = ", ".join(sorted(table_source_registry))
    raise LookupError(f"Cannot infer the table format of {str(path)!r}; declare one of: {known}.")


def source_for(paths: str | Path | Sequence[str | Path], *, format: str | None = None, **reader: Any) -> TableSource:
    """One source over one or several files of the same format."""
    listed = [paths] if isinstance(paths, str | Path) else list(paths)
    if not listed:
        raise ValueError("A source needs at least one path.")
    factory: Callable[..., TableSource] = table_source_registry.get(format or format_of(listed[0]))
    return factory(listed, **reader)


class FileSource(TableSource, ABC):
    """Several files of one format, concatenated in the declared order; reader keywords forward to pandas.

    A subclass says which suffixes imply it and how one file is read; ``format_of`` finds it by the first.
    """

    suffixes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, paths: Sequence[str | Path], **reader: Any) -> None:
        self.paths = [Path(path) for path in paths]
        self.reader = reader

    def read(self) -> Table:
        frames = [self._read_file(path) for path in self.paths]
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    @abstractmethod
    def _read_file(self, path: Path) -> Table:
        raise NotImplementedError


@table_source_registry.register("csv")
class CsvSource(FileSource):
    suffixes: ClassVar[tuple[str, ...]] = (".csv",)

    def _read_file(self, path: Path) -> Table:
        return pd.read_csv(path, **self.reader)


@table_source_registry.register("json")
class JsonSource(FileSource):
    suffixes: ClassVar[tuple[str, ...]] = (".json",)

    def _read_file(self, path: Path) -> Table:
        return pd.read_json(path, **self.reader)


@table_source_registry.register("jsonl")
class JsonLinesSource(FileSource):
    suffixes: ClassVar[tuple[str, ...]] = (".jsonl",)

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
