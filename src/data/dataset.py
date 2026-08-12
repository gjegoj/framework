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
    happen in ``__getitem__`` — that is, inside DataLoader workers — and they sit
    on opposite sides of the transform: every column is *loaded* before it, every
    target *encoded* after, so an augmentation writing a raw class name or a plain
    number hands its value to the encoder rather than overwriting one.
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
            # Every column is LOADED here, with one call whatever its kind — an input's
            # loader, or the pre-transform half of a target's encoder. A mask path
            # becomes pixels (geometry needs pixels); a plain value passes as it stands.
            inputs={name: column.loader(cells[name]) for name, column in self._schema.inputs.items()},
            targets={task: column.loader(row[column.column]) for task, column in self._schema.targets.items()},
            auxiliary_inputs={
                name: column.loader(row[column.column]) for name, column in self._schema.auxiliary_inputs.items()
            },
            meta={"row": index, Sample.CELLS: {name: cell for name, cell in cells.items() if isinstance(cell, str)}},
        )
        if self._transform is not None:
            sample = self._transform(sample)
        # Targets are ENCODED here, after the transforms — on the value an augmentation
        # may just have written, which is what makes an online target encodable at all.
        for task, column in self._schema.targets.items():
            sample.targets[task] = column.encoder.encode(sample.targets[task])
        return sample
