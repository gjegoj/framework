"""Generate a small synthetic dataset for offline smoke testing.

Creates 30 synthetic 64×64 JPEG images (in data/smoke/images/) and an annotation
CSV in data/smoke/ with target columns covering all three M2 objectives plus the
M6 embedding modality:

- ``label``        — multiclass (cat / dog / cow)
- ``binary_label`` — binary (0 / 1, e.g. is_cat)
- ``tags``         — multilabel comma-separated strings (e.g. "indoor,small")
- ``value``        — regression scalar (float)
- ``emb_path``     — path to a precomputed [D] embedding ``.npy`` (M6 modality);
                     vectors are class-shifted so the embeddings smoke can learn

Safe to re-run — overwrites existing files deterministically.
"""

import csv
import pathlib

import cv2
import numpy as np

OUT = pathlib.Path("data/smoke")
IMG_DIR = OUT / "images"
MASK_DIR = OUT / "masks"
EMB_DIR = OUT / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["cat", "dog", "cow"]
TAG_POOL = ["indoor", "outdoor", "small", "large", "fluffy"]
SEG_CLASSES = 3
EMB_DIM = 32
NUM_IMAGES = 30
RNG = np.random.default_rng(0)

rows = []
for i in range(NUM_IMAGES):
    array = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    path = IMG_DIR / f"{i}.jpg"
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
    # segmentation: single-channel index mask (classes 0..SEG_CLASSES-1)
    mask = RNG.integers(0, SEG_CLASSES, (64, 64), dtype=np.uint8)
    mask_path = MASK_DIR / f"{i}.png"
    cv2.imwrite(str(mask_path), mask)
    # embedding: class-shifted gaussian so the [D] vectors are faintly separable
    embedding = RNG.standard_normal(EMB_DIM).astype(np.float32) + LABELS.index(label)
    emb_path = EMB_DIR / f"{i}.npy"
    np.save(emb_path, embedding)
    rows.append(
        {
            "image_path": str(path),
            "label": label,
            "binary_label": binary_label,
            "tags": tags,
            "value": value,
            "mask_path": str(mask_path),
            "emb_path": str(emb_path),
        }
    )

csv_path = OUT / "data.csv"
fieldnames = ["image_path", "label", "binary_label", "tags", "value", "mask_path", "emb_path"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {NUM_IMAGES} images + {EMB_DIM}-dim embeddings → {csv_path}")
