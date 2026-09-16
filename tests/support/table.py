"""A tiny annotated table on disk: the smallest thing a run can be built from.

Shared by the tests about the composition root and by the end-to-end ones, so a run under test reads
the same kind of table a real one does — images in files, labels in a column.

The columns are the ones ``scripts/prepare_pet.py`` writes, spelled as it spells them. This table
stands in for that one wherever a test builds a shipped example, and a stand-in under other names
would have to be reached by overriding the very declarations such a test exists to check.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPECIES = ("cat", "dog")

BREEDS = {"cat": ("abyssinian", "bengal", "birman"), "dog": ("beagle", "boxer", "husky")}
"""The breeds of each species, because a breed settles a species — as it does in the real table, where
the breed lives in the file name and the species is read from it. A grouped split by breed therefore
keeps a species whole too, which is what makes one table divisible by either rule."""

MASK_BANDS = 3
"""How many classes a mask carries. The shipped segmentation examples declare three and name them, and
the names belong there; what this has to agree with is the count alone, since a pixel outside the
declared vocabulary is refused when the split is validated."""


def write_table(root: Path, samples: int = 8) -> Path:
    """Pictures of two kinds and the table naming them; returns the table's path."""
    rows = []
    for index in range(samples):
        species = SPECIES[index % 2]
        # Distinct per row and wrapped into the byte an image is made of, so any count of them fits.
        image = np.full((12, 10, 3), (20 * index + 30) % 256, dtype=np.uint8)
        cv2.imwrite(str(root / f"{index}.png"), image)
        cv2.imwrite(str(root / f"{index}_mask.png"), _bands(image.shape[:2]))
        # ``random_age`` is spelled as the real table spells it so that an example declaring that column
        # finds it, and carries no signal for the same reason — but it ascends rather than being drawn:
        # a bin layout is learned from the training split and refuses a value outside it, which noise
        # would leave to the draw. ``angle`` is upright throughout: a pretext augmentation advances it.
        # ``caption`` is written from the labels rather than collected, so a run over it proves that the
        # text half of a pipeline works and says nothing at all about reading language.
        rows.append(
            {
                "image_path": str(root / f"{index}.png"),
                "species": species,
                "breed": BREEDS[species][(index // len(SPECIES)) % len(BREEDS[species])],
                "mask_path": str(root / f"{index}_mask.png"),
                "random_age": float(index),
                "angle": 0,
                "caption": f"a {species} on the porch",
            }
        )
    table = root / "rows.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    return table


def _bands(size: tuple[int, int]) -> np.ndarray:
    """A mask naming every class it declares, in horizontal bands — the smallest dense target there is."""
    height, width = size
    return np.repeat(np.arange(height) * MASK_BANDS // height, width).reshape(height, width).astype(np.uint8)
