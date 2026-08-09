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

    Subclasses implement ``_read_file`` for a single path; multiple paths are
    concatenated in order with a fresh index. Extra keyword arguments are
    kept in ``self._reader_kwargs`` and forward verbatim to the underlying
    pandas reader — the same convention as criterion wrappers. A new format
    is a three-line subclass::

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


class LimitedSource(TableSource):
    """Another source with its rows capped — the small run you iterate on.

    Wrapping the source rather than trimming later is what keeps the meaning
    right in both dataset layouts: wrapped around one source the cap applies
    before the split (the whole run shrinks), wrapped around per-stage sources
    it applies to each stage. Both follow from where the source sits, so there
    is no rule to remember.

    Rows are drawn at random rather than taken from the top: annotation files
    routinely arrive grouped by class or ordered by date, and their first rows
    are not a sample of them. The draw has its own seed, so the subset stays
    put while everything else about the run varies.

    Parameters:
        source (TableSource): The source to read from.
        max_samples (int | float): Rows to keep. An ``int`` counts rows, a
            ``float`` in (0, 1] takes a share — the sklearn idiom, where ``1``
            means one row and ``1.0`` means all of them.
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
