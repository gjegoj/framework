"""Declarative mapping from table columns to model inputs and task targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader


@dataclass(frozen=True, slots=True)
class InputColumn:
    """One model input: which column holds it and how to load the cell value."""

    column: str
    loader: InputLoader

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("InputColumn.column must be a non-empty column name.")


@dataclass(frozen=True, slots=True)
class TargetColumn:
    """One task target: which column holds it and how to encode the raw value."""

    column: str
    encoder: TargetEncoder

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
        return input_columns | target_columns
