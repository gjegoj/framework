"""A tiny annotated table on disk: the smallest thing a run can be built from.

Shared by the tests about the composition root and by the end-to-end one, so a run under test reads
the same kind of table a real one does — pictures in files, labels in a column.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPECIES = ("cat", "dog")


def write_table(root: Path, samples: int = 8) -> Path:
    """Pictures of two kinds and the table naming them; returns the table's path."""
    rows = []
    for index in range(samples):
        # Distinct per row and wrapped into the byte a picture is made of, so any count of them fits.
        picture = np.full((12, 10, 3), (20 * index + 30) % 256, dtype=np.uint8)
        cv2.imwrite(str(root / f"{index}.png"), picture)
        # ``angle`` is upright throughout: a pretext augmentation advances it, and nothing else reads it.
        rows.append(
            {
                "image_path": str(root / f"{index}.png"),
                "species": SPECIES[index % 2],
                "age": float(index),
                "angle": 0,
            }
        )
    table = root / "rows.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    return table
