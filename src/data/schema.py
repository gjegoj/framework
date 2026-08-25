"""Declarative mapping from table columns to model inputs and task targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

# Runtime, not TYPE_CHECKING: ``Geometry.IMAGE`` is a dataclass field default, evaluated
# when this module loads.
from src.core.taxonomy import Geometry

if TYPE_CHECKING:
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader


class ColumnRole(StrEnum):
    """The three roles a table column can play in a schema, and how each is named in full.

    A column's full identity — ``input/image``, ``target/warmth`` — is a concept, not a
    string convention: assembly scopes cache namespaces with it, the warm-up titles its
    progress bars with it, and the summary reports by it. ``label`` is the one place
    the spelling exists, so no caller can misspell what it never writes.
    """

    INPUT = "input"
    AUXILIARY_INPUT = "auxiliary_input"
    TARGET = "target"

    def label(self, name: str) -> str:
        """One column's full identity: its role, a slash, its name."""
        return f"{self}/{name}"


@dataclass(frozen=True, slots=True)
class InputColumn:
    """One model input: which column holds it and how to load the cell value."""

    column: str
    loader: InputLoader
    geometry: Geometry = Geometry.IMAGE
    """How the loaded value rides augmentation geometry — light, or per-pixel labels.

    Taken from the loader's own class-level marker at build time — never written by
    hand — and read by assembly to give the column its treatment in the augmentation
    pipeline. Captured *before* any cache wrapping, because a cache wrapper is a bare
    closure and would hide the marker behind itself.
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

    def labelled_columns(self) -> list[tuple[str, InputColumn | TargetColumn]]:
        """Every column beside its full name — ``input/image``, ``target/warmth``.

        Next to ``columns()`` because it is the same walk with the names kept. The
        labels come from ``ColumnRole.label``, the same call assembly scopes cache
        namespaces with — assembly cannot use *this method* (the scoped cache is
        handed to loaders while columns are constructed, before a schema exists),
        and a test pins its role-per-section pairing to this walk.
        """
        return [
            *((ColumnRole.INPUT.label(name), column) for name, column in self.inputs.items()),
            *((ColumnRole.AUXILIARY_INPUT.label(name), column) for name, column in self.auxiliary_inputs.items()),
            *((ColumnRole.TARGET.label(name), column) for name, column in self.targets.items()),
        ]
