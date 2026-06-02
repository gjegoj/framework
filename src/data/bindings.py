"""Input and target bindings: DTOs that link ``Sample`` keys to CSV columns.

Kept in their own file so they are discoverable next to the other
single-responsibility data modules (``sources`` / ``codecs`` / ``transforms``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.codecs import TargetCodec
from src.data.loaders import InputLoader


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TargetBinding:
    """Binds a task to the data column and codec that produce its target.

    Parameters:
        name (str): Task name; also the key under which the target is stored.
        column (str): Source column in the DataFrame.
        codec (TargetCodec): Codec that decodes the raw column value.
    """

    name: str
    column: str
    codec: TargetCodec
