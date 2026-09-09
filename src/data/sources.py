"""Annotation-table sources: where the rows describing a dataset come from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.data.registry import table_source_registry

type Table = pd.DataFrame
"""The annotation-table currency of the data layer (paths, labels, metadata)."""

SUFFIX_FORMATS = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}
"""Table formats inferable from a file extension, mapped to ``table_source_registry`` keys."""


def format_of(path: str) -> str:
    """The registered format a path's suffix implies.

    Raises:
        LookupError: If the suffix implies none; lists the registered formats.
    """
    suffix = Path(path).suffix.lower()
    try:
        return SUFFIX_FORMATS[suffix]
    except KeyError:
        known = ", ".join(sorted(str(key) for key in table_source_registry))
        raise LookupError(
            f"Cannot infer the table format of '{path}'. "
            f"Give the source a 'format' explicitly; registered formats: {known}."
        ) from None


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


CAP_SEED: Final = 42
"""The draw seed of a capped table: a cap is a debugging aid, and the rows it keeps must not move between runs."""


# PYI041 reads 'int | float' as a redundant union; here it is the contract itself: a count, or a share.
def refuse_a_bad_cap(max_samples: int | float | None) -> None:  # noqa: PYI041
    """A cap that keeps nothing, or a share read as a count, refused where the cap is declared."""
    if max_samples is None:
        return
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    if isinstance(max_samples, float) and max_samples > 1.0:
        raise ValueError(f"A fractional max_samples must be at most 1.0, got {max_samples}; use a count instead.")


def capped(table: Table, max_samples: int | float | None) -> Table:  # noqa: PYI041
    """The table's rows capped — the small run you iterate on; ``None`` keeps them all.

    An ``int`` counts rows, a ``float`` in (0, 1] takes a share — the sklearn idiom. Rows
    are drawn at random rather than taken from the top: annotation files arrive grouped by
    class or date, so their first rows are not a sample.
    """
    if max_samples is None:
        return table
    if isinstance(max_samples, float):
        kept = table.sample(frac=max_samples, random_state=CAP_SEED)
    else:
        kept = table.sample(n=min(max_samples, len(table)), random_state=CAP_SEED)
    return kept.reset_index(drop=True)
