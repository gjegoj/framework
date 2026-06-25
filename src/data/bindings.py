"""Input and target bindings: DTOs that link ``Sample`` keys to CSV columns.

Kept in their own file so they are discoverable next to the other
single-responsibility data modules (``sources`` / ``encoders`` / ``transforms``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.encoders import TargetEncoder
from src.data.loaders import InputLoader


@dataclass(frozen=True, slots=True)
class InputBinding:
    """Binds a ``Sample.inputs`` key to its CSV column and loader.

    Parameters:
        name (str): Key in ``Sample.inputs`` (e.g. ``"image"``, ``"left_image"``).
        column (str): Source column in the DataFrame.
        loader (InputLoader): Loader that converts the raw value to a model input.
    """

    name: str
    column: str
    loader: InputLoader


@dataclass(frozen=True, slots=True)
class TargetBinding:
    """Binds a task to the data column and encoder that produce its target.

    Parameters:
        name (str): Task name; also the key under which the target is stored.
        column (str | None): Source column in the DataFrame, or ``None`` for a
            target-less task (e.g. triplet/contrastive) whose encoder needs no column.
        encoder (TargetEncoder): Encoder that turns the raw column value into a tensor.
    """

    name: str
    column: str | None
    encoder: TargetEncoder
