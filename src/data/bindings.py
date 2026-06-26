"""Input and target bindings: DTOs that link ``Sample`` keys to CSV columns.

Kept in their own file so they are discoverable next to the other
single-responsibility data modules (``sources`` / ``encoders`` / ``transforms``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.encoders import TargetEncoder
from src.data.loaders import InputLoader

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.core.enums import Stage
    from src.data.sources import DataSource
    from src.transforms.sample import Transform


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


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Binds a data source to the per-stage transforms applied to its samples (split mode).

    The source-side counterpart of ``InputBinding``/``TargetBinding``: where those bind a
    ``Sample`` key to its loader/encoder, this binds one source to its own augmentation
    pipeline. ``transforms`` is already resolved (the source's per-stage override merged
    with the global stage transform as fallback), so the ``DataModule`` builds one
    ``Dataset`` per source with ``transform=transforms[stage]`` and combines them with
    ``ConcatDataset``. A single-source / no-override run is one binding whose ``transforms``
    are the global ones.

    Parameters:
        source (DataSource): The built source for this group.
        transforms (Mapping[Stage, Transform]): Resolved per-stage transforms.
    """

    source: DataSource
    transforms: Mapping[Stage, Transform]
