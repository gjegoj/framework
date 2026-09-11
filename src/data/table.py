"""Annotation tables: rows bound to named inputs and targets, divided into splits, served as prepared samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from src.core import DatasetInfo, Sample
from src.data.base import DataModule, Preprocessor, Table, TableSource
from src.data.registry import data_module_registry
from src.data.sources import capped, source_for
from src.data.split import Split, split_table
from src.transforms import SampleTransform

type Source = TableSource | Table
type DeclaredSource = Source | Mapping[str, Source]


class TableDataset(Dataset[Sample]):
    """One split's rows through the preprocessor, with that split's transform between load and encode."""

    def __init__(
        self,
        table: Table,
        preprocessor: Preprocessor,
        inputs: Mapping[str, str],
        targets: Mapping[str, str],
        transform: SampleTransform | None = None,
    ) -> None:
        self.table = table
        self.preprocessor = preprocessor
        self.inputs = dict(inputs)
        self.targets = dict(targets)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> Sample:
        return self.preprocessor.preprocess(raw_sample(self.table, index, self.inputs, self.targets), self.transform)


def raw_sample(table: Table, index: int, inputs: Mapping[str, str], targets: Mapping[str, str]) -> Sample:
    """One row as the cells its names bind to; string cells stay in metadata for pages that show a source."""
    row = table.iloc[index]
    cells = {name: row[column] for name, column in inputs.items()}
    return Sample(
        inputs=cells,
        targets={name: row[column] for name, column in targets.items()},
        metadata={"row": int(index), "cells": {name: cell for name, cell in cells.items() if isinstance(cell, str)}},
    )


@data_module_registry.register("table")
class TableDataModule(DataModule):
    """Sources → (cap →) splits → fitted encoders → per-split datasets.

    ``source`` is one table (or file source) the ``split`` divides, or a mapping of already
    divided splits, in which case no ``split`` is declared. ``inputs`` and ``targets`` bind
    sample names to columns; the preprocessor's encoders read those cells.
    """

    def __init__(
        self,
        source: DeclaredSource | str | Mapping[str, Any],
        inputs: Mapping[str, str | Mapping[str, str]],
        targets: Mapping[str, str],
        *,
        preprocessor: Preprocessor,
        split: Split | Mapping[str, Any] | None = None,
        transforms: Mapping[str, SampleTransform] | None = None,
        max_samples: int | float | None = None,
    ) -> None:
        read = _sources(source)
        bound = {name: _column(name, one) for name, one in inputs.items()}
        divided_by = Split(**split) if isinstance(split, Mapping) else split
        divided = isinstance(read, Mapping)
        if divided and divided_by is not None:
            raise ValueError(
                "Per-split sources are already divided; drop the split, or declare one source for it to divide."
            )
        if not divided and divided_by is None:
            raise ValueError("One source has to be divided: declare a split with fractions, or per-split sources.")
        self._source = read
        self._inputs = bound
        self._targets = dict(targets)
        self._preprocessor = preprocessor
        self._split = divided_by
        self._transforms = dict(transforms or {})
        self._max_samples = max_samples
        self._tables: dict[str, Table] = {}

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def info(self) -> DatasetInfo:
        base = self._preprocessor.info
        return DatasetInfo(inputs=base.inputs, targets=base.targets, splits=tuple(self._tables), metadata=base.metadata)

    def setup(self, splits: Sequence[str]) -> None:
        available = self._read(splits)
        missing = sorted(set(splits) - set(available))
        if missing:
            raise LookupError(f"Splits {missing} are not available; the sources yield {sorted(available)}.")
        self._tables = {name: available[name] for name in splits}
        columns = set(self._inputs.values()) | set(self._targets.values())
        for name, table in self._tables.items():
            if absent := sorted(columns - set(map(str, table.columns))):
                raise ValueError(f"Split {name!r} lacks columns {absent}; it has {sorted(map(str, table.columns))}.")

    def fit_preprocessing(self, train_split: str) -> None:
        self._preprocessor.fit(self._cells(self._table_of(train_split)))
        for split, table in self._tables.items():
            if split == train_split:
                continue
            for name, column in self._targets.items():
                try:
                    self._preprocessor.validate({name: table[column]})
                except (LookupError, ValueError, TypeError) as error:
                    raise type(error)(f"Split {split!r}, target {name!r}: {error}") from error

    def _cells(self, table: Table) -> dict[str, pd.Series]:
        return {name: table[column] for name, column in self._targets.items()}

    def warm(self, splits: Sequence[str]) -> None:
        for split in splits:
            if split in self._tables:
                table = self._tables[split]
                rows = (raw_sample(table, i, self._inputs, self._targets) for i in range(len(table)))
                self._preprocessor.warm(rows, label=split)

    def dataset(self, split: str) -> TableDataset:
        return TableDataset(
            self._table_of(split), self._preprocessor, self._inputs, self._targets, self._transforms.get(split)
        )

    def _table_of(self, split: str) -> Table:
        try:
            return self._tables[split]
        except KeyError:
            raise LookupError(f"Split {split!r} was not set up; prepared splits: {sorted(self._tables)}.") from None

    def _read(self, splits: Sequence[str]) -> dict[str, Table]:
        """Pre-divided sources are read only for the requested splits; one source is divided whole."""
        if isinstance(self._source, Mapping):
            return {
                name: capped(_rows(source), self._max_samples)
                for name, source in self._source.items()
                if name in splits
            }
        assert self._split is not None
        return split_table(capped(_rows(self._source), self._max_samples), self._split)


def _rows(source: Source) -> Table:
    return source if isinstance(source, pd.DataFrame) else source.read()


def _sources(declared: Any) -> Source | dict[str, Source]:
    """A path, a mapping of already divided splits, or a source already built — all arrive as sources."""
    if isinstance(declared, TableSource | pd.DataFrame):
        return declared
    if isinstance(declared, Mapping) and "path" not in declared:
        return {str(name): _source(entry) for name, entry in declared.items()}
    return _source(declared)


def _source(declared: Any) -> Source:
    if isinstance(declared, TableSource | pd.DataFrame):
        return declared
    if isinstance(declared, Mapping):
        return source_for(declared["path"], format=declared.get("format"))
    return source_for(declared)


def _column(name: str, binding: Any) -> str:
    """A binding names a column, plainly or under ``column``; anything else could not be read."""
    if isinstance(binding, Mapping):
        return str(binding["column"])
    if isinstance(binding, str):
        return binding
    raise ValueError(f"Input {name!r} binds to a column name or {{column: ...}}, got {binding!r}.")
