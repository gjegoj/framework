"""A detection dataset in the framework's own canon, small enough to build in a test."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.config import load_config
from src.data.converters.annotations import annotation_object, annotation_row, write_annotations

if TYPE_CHECKING:
    from pathlib import Path

CLASSES = {0: "cat", 1: "dog"}
DOG = [annotation_object((50.0, 25.0, 150.0, 75.0), "dog")]
"""The measured box: in a 200x100 picture letterboxed to 64x64 it lands on [16, 24, 48, 40]."""
CAT = [annotation_object((0.0, 0.0, 40.0, 40.0), "cat")]


def annotation_tree(root: Path) -> None:
    """Three train images — one dog, one cat, one negative — and one val image.

    Written by the canon writer itself, so this is the writer-to-reader path a real run
    takes; the on-disk spelling is pinned by the reader's own test and the converters'.
    """
    for name in ("a", "b", "c", "d"):
        cv2.imwrite(str(root / f"{name}.jpg"), np.full((100, 200, 3), 128, dtype=np.uint8))
    write_annotations(
        [annotation_row("a.jpg", DOG), annotation_row("b.jpg", CAT), annotation_row("c.jpg", [])], root / "train.jsonl"
    )
    write_annotations([annotation_row("d.jpg", DOG)], root / "val.jsonl")


def detection_config(root: Path, **sections: Any) -> Any:
    """What a detection run declares: per-stage sources, an image column, the kind and its target."""
    stage_pipeline = {
        "_target_": "src.transforms.AlbumentationsTransform",
        "transforms": [
            {"_target_": "albumentations.LetterBox", "size": [64, 64]},
            {"_target_": "albumentations.Normalize"},
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ],
    }
    return load_config(
        {
            "data": {
                "source": {"train": str(root / "train.jsonl"), "val": str(root / "val.jsonl")},
                "inputs": {"image": {"column": "image", "loader": {"name": "image", "root": str(root)}}},
            },
            "tasks": {"boxes": {"kind": "detection", "target": "objects", "classes": CLASSES}},
            "model": {"name": "timm", "model_name": "resnet18", "pretrained": False},
            "transforms": {stage: dict(stage_pipeline) for stage in ("train", "val")},
        }
        | sections
    )
