"""Declarative mapping from table columns to model inputs and task targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader


@dataclass(frozen=True, slots=True)
class InputColumn:
    """One model input: which column holds it and how to load the cell value."""

    column: str
    loader: InputLoader
    spatial: bool = False
    """Whether the loaded value is per-pixel labels rather than light.

    Taken from the loader's own class-level marker at build time — never written by
    hand — and read by assembly to give the column mask treatment in the augmentation
    pipeline: nearest-neighbour geometry, and ``Normalize`` leaving it alone. Captured
    *before* any cache wrapping, because a cache wrapper is a bare closure and would
    hide the marker behind itself.
    """

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("InputColumn.column must be a non-empty column name.")


@dataclass(frozen=True, slots=True)
class TargetColumn:
    """One task target: which column holds it and how it becomes training data."""

    column: str
    encoder: TargetEncoder

    @property
    def loader(self) -> InputLoader:
        """The encoder's pre-transform half, in the shape every column's loader has.

        A bound ``load`` *is* an ``InputLoader`` — one table cell in, its raw form out —
        so the dataset and the cache warm every kind of column with one call, whatever
        the column is.
        """
        return self.encoder.load

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("TargetColumn.column must be a non-empty column name.")


@dataclass(frozen=True, slots=True)
class DataSchema:
    """The table contract: named inputs and per-task targets.

    Keys of ``inputs`` name model inputs (they become ``Batch.inputs``); keys
    of ``targets`` are task names (they become ``Batch.targets``). Targets may
    be empty — structure-only tasks (metric learning) have no target column.
    """

    inputs: Mapping[str, InputColumn]
    targets: Mapping[str, TargetColumn]
    auxiliary_inputs: Mapping[str, InputColumn] = field(default_factory=dict)
    """Columns the augmentations read and nothing downstream sees.

    Loaded exactly as ``inputs`` are, and modelled by the same ``InputColumn`` — the
    difference is where they go afterwards, which is nowhere: no ``Batch`` slot exists
    for them.
    """

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError(
                "A table needs at least one input column: 'data.inputs' maps a model input "
                "name to the column holding it, e.g. inputs: {image: {column: image_path}}."
            )

    def columns(self) -> set[str]:
        """Every table column the schema references — used for fail-fast validation."""
        input_columns = {input_column.column for input_column in self.inputs.values()}
        target_columns = {target_column.column for target_column in self.targets.values()}
        auxiliary_columns = {input_column.column for input_column in self.auxiliary_inputs.values()}
        return input_columns | target_columns | auxiliary_columns
