"""Annotation-table sources: where the rows describing a dataset come from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.registry import table_source_registry

type Table = pd.DataFrame
"""The annotation-table currency of the data layer (paths, labels, metadata)."""


class TableSource(ABC):
    """Reads the annotation table describing a dataset.

    Implementations own one storage format each — CSV, parquet, a database,
    a COCO json flattened to rows. Everything downstream sees only ``Table``.
    """

    @abstractmethod
    def read(self) -> Table:
        """Load the full annotation table."""


class FileSource(TableSource):
    """Base for file-backed sources: one format per subclass.

    Subclasses implement ``_read_file`` for one path; several paths are concatenated in
    order. Extra keyword arguments forward verbatim to the pandas reader::

        @table_source_registry.register("parquet")
        class ParquetSource(FileSource):
            def _read_file(self, path: Path) -> Table:
                return pd.read_parquet(path, **self._reader_kwargs)

    Parameters:
        paths (str | Path | Sequence): One file path or several, in order.
        **kwargs: Forwarded verbatim to the pandas reader of the format.
    """

    def __init__(self, paths: str | Path | Sequence[str | Path], **kwargs: Any) -> None:
        entries = [paths] if isinstance(paths, str | Path) else list(paths)
        if not entries:
            raise ValueError("FileSource needs at least one path.")
        self._paths = [Path(entry) for entry in entries]
        self._reader_kwargs = kwargs

    def read(self) -> Table:
        frames = [self._read_file(path) for path in self._paths]
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    @abstractmethod
    def _read_file(self, path: Path) -> Table:
        """Read one file of this source's format."""


@table_source_registry.register("csv")
class CsvSource(FileSource):
    """CSV annotation tables via ``pd.read_csv`` (``sep``, ``dtype``, ... via kwargs)."""

    def _read_file(self, path: Path) -> Table:
        return pd.read_csv(path, **self._reader_kwargs)


@table_source_registry.register("json")
class JsonSource(FileSource):
    """JSON annotation tables via ``pd.read_json`` (``orient``, ``lines``, ... via kwargs)."""

    def _read_file(self, path: Path) -> Table:
        return pd.read_json(path, **self._reader_kwargs)


@table_source_registry.register("jsonl")
class JsonLinesSource(FileSource):
    """JSON Lines tables — one row per line, nested values kept as they are written.

    Its own source rather than ``json`` with ``lines: true``, so ``.jsonl`` is inferable
    from the suffix and a row's nested annotations need no declaration.
    """

    def _read_file(self, path: Path) -> Table:
        return pd.read_json(path, lines=True, **self._reader_kwargs)


class LimitedSource(TableSource):
    """Another source with its rows capped — the small run you iterate on.

    Wrapping the source keeps the meaning right in both layouts: around one source the cap
    applies before the split, around per-stage sources to each stage. Rows are drawn at
    random with their own seed — annotation files arrive grouped by class or date, so their
    first rows are not a sample.

    Parameters:
        source (TableSource): The source to read from.
        max_samples (int | float): Rows to keep. An ``int`` counts rows, a ``float`` in
            (0, 1] takes a share — the sklearn idiom.
        seed (int): Draw seed; the same seed always keeps the same rows.
    """

    # PYI041 reads 'int | float' as a redundant union; here it is the contract itself —
    # the two halves mean different things, and 'float' alone would hide that from a caller.
    def __init__(self, source: TableSource, max_samples: int | float, seed: int = 42) -> None:  # noqa: PYI041
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}.")
        if isinstance(max_samples, float) and max_samples > 1.0:
            raise ValueError(f"A fractional max_samples must be at most 1.0, got {max_samples}; use a count instead.")
        self._source = source
        self._max_samples = max_samples
        self._seed = seed

    def read(self) -> Table:
        table = self._source.read()
        if isinstance(self._max_samples, float):
            kept = table.sample(frac=self._max_samples, random_state=self._seed)
        else:
            kept = table.sample(n=min(self._max_samples, len(table)), random_state=self._seed)
        return kept.reset_index(drop=True)


class InMemorySource(TableSource):
    """A table handed in directly — notebooks, tests, generated data.

    Not registry-listed: a live ``DataFrame`` is not expressible from config.
    """

    def __init__(self, table: Table) -> None:
        self._table = table

    def read(self) -> Table:
        return self._table
