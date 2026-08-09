"""Prepare the Oxford-IIIT Pet dataset as one table this framework can train on.

Downloads the dataset through torchvision — no account, no API key — remaps the
trimaps to 0-based class indices, and writes a CSV with four real columns and one
synthetic one:

    image_path  the picture                                    (input)
    species     cat / dog                                      (GLOBAL binary)
    breed       one of 37                                      (GLOBAL multiclass)
    mask_path   pet / background / boundary                    (DENSE segmentation)
    random_age  noise, by that name                            (GLOBAL regression)

``random_age`` is noise on purpose, and named so nobody reads it as a learnable
target: no model can beat predicting its mean, because nothing in the picture
carries it. That makes it a test of the *pipeline* — does a regression task build,
train, log and report — rather than of a model. A ``mae`` that settles near the
column's own standard deviation is the correct outcome, not a failure.

The whole dataset is written: 7349 rows from ``annotations/list.txt``. The
reference this is modelled on read ``trainval.txt`` instead and used 3680 of them,
without saying so.

``torchvision`` is not among this project's declared dependencies — it arrives
under torch. That is tolerable for a one-off script in a way it would not be for
library code, and it is said here rather than left to be discovered.

Run: uv run python scripts/prepare_pet.py
"""

from __future__ import annotations

import argparse
import csv
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPECIES: dict[int, str] = {1: "cat", 2: "dog"}
"""What the index file's species column means, per its own header."""

AGE_RANGE: tuple[float, float] = (0.5, 15.0)
"""Plausible years for a pet, so the column reads like data even though it is not."""

_TRIMAP_CLASSES = 3
"""pet, background, boundary — the three the remapped mask holds."""


@dataclass(frozen=True, slots=True)
class Listed:
    """One row of the dataset's own index, before its files are known to be readable."""

    name: str
    species: str

    @property
    def breed(self) -> str:
        """The breed, which lives only in the file name.

        The index carries a numeric class id and no vocabulary to resolve it
        against, so the name is the only place the breed is spelled.
        """
        return self.name.rsplit("_", 1)[0]


def listed(annotations: Path) -> list[Listed]:
    """Every image the dataset indexes — all 7349, not one split of it.

    ``list.txt`` is the whole dataset; ``trainval.txt`` and ``test.txt`` are its
    halves. Splitting is this framework's job and it does it from config, so the
    table it is handed should be everything there is.
    """
    rows: list[Listed] = []
    for line in (annotations / "list.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, _class_id, species_id, _breed_id = line.split()
        rows.append(Listed(name=name, species=SPECIES[int(species_id)]))
    return rows


def zero_based(trimap: np.ndarray) -> np.ndarray:
    """Oxford's trimap holds ``{1, 2, 3}``; a cross-entropy target starts at zero."""
    return (trimap.astype(np.int16) - 1).clip(0, _TRIMAP_CLASSES - 1).astype(np.uint8)


def random_age(name: str, seed: int) -> float:
    """A plausible age, drawn per row and reproducibly — and carrying no signal.

    Keyed by the image's own name rather than by its position in the file, so a row
    keeps its age through a reordering, a filter, or a re-run over a different
    subset. Position-keyed noise would quietly relabel the dataset the first time a
    row was dropped, and two runs of the same script would disagree about what they
    trained on.
    """
    draw = np.random.default_rng(zlib.crc32(f"{seed}:{name}".encode()))
    return round(float(draw.uniform(*AGE_RANGE)), 1)


def prepared(
    rows: list[Listed], images: Path, trimaps: Path, masks_out: Path, seed: int
) -> tuple[list[dict[str, str]], Counter[str]]:
    """The records to write, and a count of what was left out and why.

    An image is verified by decoding it with the same OpenCV the framework's own
    ``ImageLoader`` uses, so "readable here" means "readable there". The reference
    verified with a different library than the one that would later read the file,
    and wrapped the attempt in a bare ``except Exception`` — which reads a
    permissions error as "skip this image".
    """
    records: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        image_path, trimap_path = images / f"{row.name}.jpg", trimaps / f"{row.name}.png"
        if not image_path.exists() or not trimap_path.exists():
            skipped["missing file"] += 1
            continue
        # `imread` answers None for anything it cannot decode, so a corrupt file is
        # a value to check rather than an exception to catch and guess at.
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            skipped["unreadable image"] += 1
            continue
        trimap = cv2.imread(str(trimap_path), cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            skipped["unreadable mask"] += 1
            continue
        mask_path = masks_out / f"{row.name}.png"
        cv2.imwrite(str(mask_path), zero_based(trimap))
        records.append(
            {
                "image_path": str(image_path.resolve()),
                "species": row.species,
                "breed": row.breed,
                "mask_path": str(mask_path.resolve()),
                "random_age": str(random_age(row.name, seed)),
            }
        )
    return records, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/pet/raw", help="where torchvision downloads to")
    parser.add_argument("--out", default="data/pet", help="where the masks and the table are written")
    parser.add_argument("--seed", type=int, default=0, help="fixes random_age; the same seed rebuilds the same table")
    arguments = parser.parse_args()

    import torchvision  # lazy: needed for the one-time download and nothing else

    print("Downloading Oxford-IIIT Pet through torchvision (~800 MB, once)…", flush=True)
    torchvision.datasets.OxfordIIITPet(
        root=arguments.root, split="trainval", target_types="segmentation", download=True
    )

    base = Path(arguments.root) / "oxford-iiit-pet"
    annotations = base / "annotations"
    masks_out = Path(arguments.out) / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    rows = listed(annotations)
    records, skipped = prepared(rows, base / "images", annotations / "trimaps", masks_out, arguments.seed)
    if not records:
        raise SystemExit(f"Nothing to write: all {len(rows)} indexed images were skipped ({dict(skipped)}).")

    table = Path(arguments.out) / "data.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # What was left out is reported beside what was written: a preparation script
    # whose output cannot be reconciled with its input is one nobody can trust.
    species = Counter(record["species"] for record in records)
    ages = [float(record["random_age"]) for record in records]
    print(
        f"Wrote {table}: {len(records)} of {len(rows)} rows "
        f"({species['cat']} cat / {species['dog']} dog, "
        f"{len({record['breed'] for record in records})} breeds). "
        f"Masks in {masks_out}."
    )
    if skipped:
        print(f"Skipped {sum(skipped.values())}: " + ", ".join(f"{count} {why}" for why, count in skipped.items()))
    print(
        f"random_age spans {min(ages)}–{max(ages)} with a deviation of {float(np.std(ages)):.2f}. "
        f"It is noise: a regression on it cannot do better than that number, and doing so would mean a leak."
    )


if __name__ == "__main__":
    main()
