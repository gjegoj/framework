"""A torch ``Dataset`` assembling ``Sample``s from table rows via a ``DataSchema``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from src.core.entities import Sample
from src.data.schema import DataSchema
from src.data.sources import Table

if TYPE_CHECKING:
    from src.core.ports import SampleTransform


class TableDataset(Dataset[Sample]):
    """Materializes one ``Sample`` per table row.

    The schema is validated against the table upfront, so a typo in a column
    name fails at construction rather than mid-epoch. Loading and encoding
    happen in ``__getitem__`` — that is, inside DataLoader workers.
    """

    def __init__(self, table: Table, schema: DataSchema, transform: SampleTransform | None = None) -> None:
        missing = sorted(schema.columns() - set(table.columns))
        if missing:
            raise ValueError(f"Schema references columns missing from the table: {', '.join(missing)}.")
        self._table = table
        self._schema = schema
        self._transform = transform

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, index: int) -> Sample:
        row = self._table.iloc[index]
        cells = {name: row[input_column.column] for name, input_column in self._schema.inputs.items()}
        sample = Sample(
            inputs={name: input_column.loader(cells[name]) for name, input_column in self._schema.inputs.items()},
            targets={
                task: target_column.encoder.encode(row[target_column.column])
                for task, target_column in self._schema.targets.items()
            },
            # A string cell is readable as it stands — a path, a URL, a caption. An
            # array is not, and the tensor built from it already went to the model.
            # Carry the readable ones and let the consumer decide their use: a report
            # links a path and prints a caption, and only it knows which it wanted.
            meta={"row": index, Sample.CELLS: {name: cell for name, cell in cells.items() if isinstance(cell, str)}},
        )
        return self._transform(sample) if self._transform is not None else sample
