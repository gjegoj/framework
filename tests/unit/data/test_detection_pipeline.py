"""A table-declared detection dataset: canon rows in, letterboxed ``Instances`` batches out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.assembly.data import build_data_module
from src.config import load_config
from src.core import DataProfile, Instances, Stage
from src.data.collate import collate_samples
from src.data.converters.canon import canon_object, canon_record, write_canon

DOG = [canon_object((50.0, 25.0, 150.0, 75.0), "dog")]
"""The measured box: in a 200x100 picture letterboxed to 64x64 it lands on [16, 24, 48, 40]."""

CAT = [canon_object((0.0, 0.0, 40.0, 40.0), "cat")]


def canon_tree(root: Path) -> None:
    """Three train images — one dog, one cat, one negative — and one val image.

    Written by the canon writer itself, so this is the writer-to-reader path a real run
    takes; the on-disk spelling is pinned by the reader's own test and the converters'.
    """
    for name in ("a", "b", "c", "d"):
        cv2.imwrite(str(root / f"{name}.jpg"), np.full((100, 200, 3), 128, dtype=np.uint8))
    write_canon(
        [canon_record("a.jpg", DOG), canon_record("b.jpg", CAT), canon_record("c.jpg", [])], root / "train.jsonl"
    )
    write_canon([canon_record("d.jpg", DOG)], root / "val.jsonl")


def detection_config(root: Path) -> Any:
    """What a detection run declares: a source, an image column, a preset and a target."""
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
            "tasks": {"boxes": {"preset": "detection", "target": "objects"}},
            "model": {"name": "timm", "model_name": "resnet18", "pretrained": False},
            "transforms": {stage: dict(stage_pipeline) for stage in ("train", "val")},
        }
    )


def test_canon_rows_become_letterboxed_instances_batches(tmp_path: Path) -> None:
    """Every seam of the stage at once: source, encoder, geometry, collate — one assertion each."""
    canon_tree(tmp_path)
    module = build_data_module(detection_config(tmp_path))
    profile = DataProfile()
    module.setup(profile)

    loader = DataLoader(module.dataset(Stage.TRAIN), batch_size=3, collate_fn=module.collate or collate_samples)
    batch = next(iter(loader))
    merged = batch.targets["boxes"]

    assert profile.facts("boxes").num_classes == 2
    assert profile.facts("boxes").class_names == ["cat", "dog"]
    assert isinstance(merged, Instances)
    assert batch.inputs["image"].shape == (3, 3, 64, 64)


def test_the_boxes_of_a_batch_carry_the_letterbox_arithmetic_end_to_end(tmp_path: Path) -> None:
    """The number measured on the seam, now through encoder, transform and collation."""
    canon_tree(tmp_path)
    module = build_data_module(detection_config(tmp_path))
    module.setup(DataProfile())

    loader = DataLoader(module.dataset(Stage.VAL), batch_size=1, collate_fn=module.collate or collate_samples)
    merged = next(iter(loader)).targets["boxes"]

    assert isinstance(merged, Instances)
    assert torch.allclose(merged.boxes, torch.tensor([[16.0, 24.0, 48.0, 40.0]]))
    assert merged.labels.tolist() == [1]  # "dog" is second in the learned vocabulary


def test_a_negative_row_takes_its_place_in_the_batch_holding_nothing(tmp_path: Path) -> None:
    """The row without objects is an image the model must learn to leave empty."""
    canon_tree(tmp_path)
    module = build_data_module(detection_config(tmp_path))
    module.setup(DataProfile())

    dataset = module.dataset(Stage.TRAIN)
    batch = collate_samples([dataset[index] for index in range(3)])
    merged = batch.targets["boxes"]

    assert isinstance(merged, Instances)
    assert len(merged.of(2).boxes) == 0
    assert merged.sample_index.tolist() == [0, 1]
