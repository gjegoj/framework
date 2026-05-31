"""Generate a small synthetic dataset for offline smoke testing.

Creates 30 synthetic 64×64 JPEG images and an annotation CSV in data/smoke/
with four target columns covering all three new M2 objectives:

- ``label``        — multiclass (cat / dog / cow)
- ``binary_label`` — binary (0 / 1, e.g. is_cat)
- ``tags``         — multilabel comma-separated strings (e.g. "indoor,small")
- ``value``        — regression scalar (float)

Safe to re-run — overwrites existing files deterministically.
"""

import csv
import pathlib

import cv2
import numpy as np

OUT = pathlib.Path("data/smoke")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["cat", "dog", "cow"]
TAG_POOL = ["indoor", "outdoor", "small", "large", "fluffy"]
NUM_IMAGES = 30
RNG = np.random.default_rng(0)

rows = []
for i in range(NUM_IMAGES):
    array = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    path = OUT / f"{i}.jpg"
    cv2.imwrite(str(path), array)
    label = LABELS[i % len(LABELS)]
    # binary: is it a cat? (1 = yes)
    binary_label = 1 if label == "cat" else 0
    # multilabel: 2-3 random tags from pool, comma-separated, deterministic
    n_tags = 2 + (i % 2)
    tag_indices = [(i + j) % len(TAG_POOL) for j in range(n_tags)]
    tags = ",".join(TAG_POOL[idx] for idx in sorted(set(tag_indices)))
    # regression: synthetic float derived from index
    value = round(float(RNG.uniform(0.0, 10.0)), 4)
    rows.append(
        {
            "image_path": str(path),
            "label": label,
            "binary_label": binary_label,
            "tags": tags,
            "value": value,
        }
    )

csv_path = OUT / "data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label", "binary_label", "tags", "value"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {NUM_IMAGES} images → {csv_path}")
