"""Prepare an Oxford-IIIT Pet subset for the multitask training smoke.

Downloads the dataset via torchvision (reliable, no auth), takes a random subset,
remaps the trimap masks {1,2,3} -> {0,1,2} (CrossEntropy expects 0-based indices),
and writes a CSV the framework's CsvDataSource can read:

    image_path, species, breed, mask_path

- species  : "cat" / "dog"   (GLOBAL binary classification)
- breed    : breed name       (GLOBAL multiclass, optional task)
- mask_path: remapped trimap  (DENSE segmentation, 3 classes: pet / background / boundary)

Run: uv run python scripts/prepare_pet.py --n 240
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_trainval(annotations: Path) -> list[tuple[str, int]]:
    """Read ``annotations/trainval.txt`` → ``[(image_stem, species_id), ...]``.

    Line format: ``<name> <class_id> <species_id> <breed_id>`` where species 1=Cat, 2=Dog.
    """
    rows: list[tuple[str, int]] = []
    for line in (annotations / "trainval.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, _class_id, species_id, _breed_id = line.split()
        rows.append((name, int(species_id)))
    return rows


def _verify_rgb(path: Path) -> bool:
    """A few Oxford images are corrupt / non-RGB; keep only ones that load cleanly."""
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/pet/raw", help="torchvision download root")
    parser.add_argument("--out", default="data/pet", help="output dir for masks + csv")
    parser.add_argument("--n", type=int, default=240, help="subset size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torchvision  # lazy: only needed for the one-time download

    print("Downloading Oxford-IIIT Pet via torchvision (≈800MB, one-time)…", flush=True)
    torchvision.datasets.OxfordIIITPet(root=args.root, split="trainval", target_types="segmentation", download=True)

    base = Path(args.root) / "oxford-iiit-pet"
    images_dir, annotations = base / "images", base / "annotations"
    trimaps = annotations / "trimaps"

    out = Path(args.out)
    masks_out = out / "masks"
    masks_out.mkdir(parents=True, exist_ok=True)

    rows = _parse_trainval(annotations)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)

    records: list[dict[str, str]] = []
    for name, species_id in rows:
        if len(records) >= args.n:
            break
        image_path = images_dir / f"{name}.jpg"
        trimap_path = trimaps / f"{name}.png"
        if not image_path.exists() or not trimap_path.exists() or not _verify_rgb(image_path):
            continue

        trimap = np.array(Image.open(trimap_path))  # values {1, 2, 3}
        remapped = (trimap.astype(np.int16) - 1).clip(0, 2).astype(np.uint8)  # → {0, 1, 2}
        mask_path = masks_out / f"{name}.png"
        Image.fromarray(remapped, mode="L").save(mask_path)

        records.append(
            {
                "image_path": str(image_path.resolve()),
                "species": "cat" if species_id == 1 else "dog",
                "breed": name.rsplit("_", 1)[0],
                "mask_path": str(mask_path.resolve()),
            }
        )

    csv_path = out / "data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "species", "breed", "mask_path"])
        writer.writeheader()
        writer.writerows(records)

    breeds = sorted({r["breed"] for r in records})
    (out / "breeds.txt").write_text("\n".join(breeds))
    n_cat = sum(r["species"] == "cat" for r in records)
    print(
        f"Wrote {csv_path} — {len(records)} samples ({n_cat} cat / {len(records) - n_cat} dog), "
        f"{len(breeds)} breeds. Masks in {masks_out}."
    )


if __name__ == "__main__":
    main()
