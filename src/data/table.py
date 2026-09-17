"""Annotation tables: rows bound to named inputs and targets, divided into splits, served as prepared samples."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from src.core import CELLS, DatasetInfo, DatasetStatistics, Distribution, Sample, Stage, naming
from src.data.base import DataModule, Preprocessor, Table, TableSource
from src.data.registry import data_module_registry
from src.data.sources import capped, source_for
from src.data.split import Split, split_table
from src.transforms import SampleTransform

log = logging.getLogger(__name__)

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
        metadata={"row": int(index), CELLS: {name: cell for name, cell in cells.items() if isinstance(cell, str)}},
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
        divided_by = Split.declared(split) if isinstance(split, Mapping) else split
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
        available = _standing_in_for_a_test_split_that_was_not_declared(self._read(splits), splits)
        missing = sorted(set(splits) - set(available))
        if missing:
            raise LookupError(f"Splits {missing} are not available; the sources yield {sorted(available)}.")
        self._tables = {name: available[name] for name in splits}
        columns = set(self._inputs.values()) | set(self._targets.values())
        for name, table in self._tables.items():
            if absent := sorted(columns - set(map(str, table.columns))):
                raise ValueError(f"Split {name!r} lacks columns {absent}; it has {sorted(map(str, table.columns))}.")

    def fit_preprocessing(self, train_split: str) -> None:
        self._read_targets(train_split, self._preprocessor.fit)
        for split in self._tables:
            if split != train_split:
                self._read_targets(split, self._preprocessor.validate)

    def _read_targets(self, split: str, read: Callable[[Mapping[str, pd.Series]], None]) -> None:
        """One target at a time, so that whatever refuses a column says which column of which split.

        An encoder knows what it reads and not where it was reading: `Values outside the declared
        classes: bird` names neither the split that held it nor the column it came from, and a run
        with several targets has several columns it could have been any of.
        """
        table = self._table_of(split)
        for name, column in self._targets.items():
            with naming(f"Split {split!r}, target {name!r}"):
                read({name: table[column]})

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

    def statistics(self) -> DatasetStatistics:
        """How many rows each prepared split holds, and what each of its target columns looks like.

        Read from the annotation tables rather than from batches, and keyed target-first because that
        is how it is read: one table per target, a column per split, so an imbalance that differs
        between train and test is a line rather than two reports to compare.
        """
        described = {split: self._preprocessor.describe(self._cells(table)) for split, table in self._tables.items()}
        targets: dict[str, dict[str, Distribution]] = {}
        for split, distributions in described.items():
            for name, distribution in distributions.items():
                targets.setdefault(name, {})[split] = distribution
        return DatasetStatistics(rows={split: len(table) for split, table in self._tables.items()}, targets=targets)

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


def _standing_in_for_a_test_split_that_was_not_declared(
    available: dict[str, Table], splits: Sequence[str]
) -> dict[str, Table]:
    """A run that will test, with no test rows of its own and a validation split to stand in for them.

    Here because this is the one place that sees both halves: which stages the run will read, and what
    the sources actually yielded — and it holds for both grammars, a ``split`` of fractions and sources
    already divided, because it is one rule about the same two facts.

    Said out loud rather than filled in quietly, because it changes what a reported number means: the
    epoch a run keeps is the one its validation split chose, so a number reported as ``test`` is that
    same measurement under another name rather than an estimate on data held out from the choice. A run
    with no validation split either is refused as before — there is nothing to stand in.
    """
    if Stage.TEST in splits and Stage.TEST not in available and Stage.VAL in available:
        log.warning(
            "No test split is declared, so the test stage reads the validation rows: what this run "
            "reports as test is measured on the rows it chose its epoch by, not on data held out from "
            "that choice. Declare a test split to report on one."
        )
        return {**available, Stage.TEST: available[Stage.VAL]}
    return available


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
    """One source: a path, the long form of one, or a source already built.

    The long form is checked here because nothing else can: it reaches a run as a plain mapping that
    no schema types, so the key beside ``path`` is either read by this line or read by nobody. A
    key that is not ``format`` left the format to be inferred from the suffix, which is exactly the job
    the key was written to take away from it — and the run then read the file the way it forbade.
    """
    if isinstance(declared, TableSource | pd.DataFrame):
        return declared
    if isinstance(declared, Mapping):
        _refuse_a_mapping_that_is_no_source(declared)
        return source_for(declared["path"], format=declared.get("format"))
    return source_for(declared)


def _refuse_a_mapping_that_is_no_source(declared: Mapping[str, Any]) -> None:
    """Both halves of the long form: every key it takes, and the one it cannot do without."""
    if unknown := sorted(declared.keys() - {"path", "format"}):
        named = ", ".join(repr(str(key)) for key in unknown)
        raise ValueError(
            f"{named} {'is' if len(unknown) == 1 else 'are'} no part of a source: `data.source` names a "
            "file, or {path: ..., format: ...} where the suffix does not name the format."
        )
    if "path" not in declared:
        raise ValueError(
            "A source names a file and this one names none: write `data.source` as a path, or as "
            "{path: ..., format: ...}. Per-split sources are a mapping of split names to those."
        )


def _column(name: str, binding: Any) -> str:
    """A binding names a column, plainly or under ``column``; anything else could not be read."""
    if isinstance(binding, Mapping):
        return str(binding["column"])
    if isinstance(binding, str):
        return binding
    raise ValueError(f"Input {name!r} binds to a column name or {{column: ...}}, got {binding!r}.")
