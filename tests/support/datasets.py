"""Files on disk a data pipeline can actually read: images, masks, and the table naming them.

Three primitives and one composition of them. A test that wants the usual
classification fixture calls ``write_dataset``; a test whose subject *is* the target
column builds its own records over ``write_images`` and ``write_table``, so what it is
about stays in view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

IMAGE_SIDE: Final = 16
"""Side of a written image. Small enough to be free, large enough for a real resize."""

TABLE_NAME: Final = "annotations.csv"
"""What the annotation table is called under a dataset root."""

MASK_CLASSES: Final = 3
"""Distinct indices a written mask cycles through."""


def write_images(root: Path, rows: int, side: int = IMAGE_SIDE) -> list[str]:
    """Write one flat-grey PNG per row and return their paths relative to ``root``.

    The grey level follows the row index, so a test can tell one sample from another
    after augmentation without carrying a fixture image around.
    """
    directory = root / "images"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(rows):
        cv2.imwrite(str(directory / f"{index}.png"), np.full((side, side, 3), index * 8 % 256, dtype=np.uint8))
        written.append(f"images/{index}.png")
    return written


def write_masks(root: Path, rows: int, classes: int = MASK_CLASSES, side: int = IMAGE_SIDE) -> list[str]:
    """Write one single-class index map per row and return their paths relative to ``root``.

    Each mask holds exactly one class, cycling through them, so a run over ``classes``
    rows or more sees every index at least once and the fitted class count is the
    declared one.
    """
    directory = root / "masks"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(rows):
        cv2.imwrite(str(directory / f"{index}.png"), np.full((side, side), index % classes, dtype=np.uint8))
        written.append(f"masks/{index}.png")
    return written


def write_table(root: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write the annotation table under ``root`` and return its path."""
    path = root / TABLE_NAME
    pd.DataFrame(list(records)).to_csv(path, index=False)
    return path


def write_dataset(
    root: Path,
    rows: int = 8,
    *,
    side: int = IMAGE_SIDE,
    masks: bool = False,
    mask_classes: int = MASK_CLASSES,
) -> Path:
    """Images, an alternating ``label`` column, and optionally masks — the usual fixture.

    Returns the table's path, which is what a ``data.source`` names. A test needing the
    rows themselves reads it back: the file is the source of truth either way.

    Parameters:
        root (Path): Where the images, masks and table are written.
        rows (int): How many samples. Keep it divisible by the split's denominators.
        side (int): Image side in pixels.
        masks (bool): Also write a per-row index map and name it in a ``mask`` column.
        mask_classes (int): How many distinct indices those masks cycle through.
    """
    images = write_images(root, rows, side)
    records: list[dict[str, Any]] = [
        {"image": image, "label": "cat" if index % 2 else "dog"} for index, image in enumerate(images)
    ]
    if masks:
        for record, mask in zip(records, write_masks(root, rows, mask_classes, side), strict=True):
            record["mask"] = mask
    return write_table(root, records)
