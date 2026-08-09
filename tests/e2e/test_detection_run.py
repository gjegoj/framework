"""End to end: a YOLO descriptor becomes a run, and what it found becomes a number.

The whole seam in one pass — a vendor dataset read through the `DataModule` port, a
vendor loss arriving as named parts, and a ragged prediction reaching a metric the
tracker can show. Nothing here mentions ultralytics: the config names a family and the
framework does the rest, which is the claim this file exists to check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
import yaml

from src.assembly import assemble, run
from src.config import load_config
from src.models import YoloModel

pytest.importorskip("ultralytics", reason="the YOLO family is an optional dependency")

CLASSES = {0: "cat", 1: "dog"}
SIZE = 64


def write_yolo_dataset(root: Path) -> str:
    """The layout ultralytics reads: images, labels beside them, and a descriptor.

    Two images per stage, each carrying one box of its own class — enough for a mAP that
    is a number rather than an empty average, and small enough to build in a test.
    """
    for stage in ("train", "val"):
        (root / "images" / stage).mkdir(parents=True, exist_ok=True)
        (root / "labels" / stage).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            picture = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
            # A bright square where the box is, so there is something to find.
            picture[16:48, 16:48] = 220
            cv2.imwrite(str(root / "images" / stage / f"{index}.jpg"), picture)
            (root / "labels" / stage / f"{index}.txt").write_text(f"{index} 0.5 0.5 0.5 0.5\n")
    descriptor = root / "data.yaml"
    descriptor.write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": CLASSES})
    )
    return str(descriptor)


@pytest.mark.e2e
@pytest.mark.slow
def test_a_detection_run_trains_and_reports_what_it_found(tmp_path: Path) -> None:
    """One epoch on four images: the losses log by name and the metric family appears.

    `train/boxes/*` are the vendor's own three components under the framework's grammar;
    `val/boxes/map/*` is the family one metric published from one pass. Neither is
    asserted for its value — four images decide nothing — but both have to exist, because
    their absence is what a broken seam looks like.
    """
    descriptor = write_yolo_dataset(tmp_path)
    config = load_config(
        {
            "image_size": [SIZE, SIZE],
            "data": {"source": descriptor, "inputs": {}},
            "tasks": {"boxes": {"preset": "detection"}},
            "model": {"name": "yolo", "model_name": "yolov8n.yaml", "mosaic": 0.0},
            "loader": {"batch_size": 2},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "logger": False,
                "default_root_dir": str(tmp_path),
                "enable_checkpointing": False,
            },
            "run": {"directory": str(tmp_path / "run"), "test": False},
        }
    )

    experiment = assemble(config)
    run(experiment, config)

    logged = set(experiment.trainer.callback_metrics)
    assert {"train/boxes/box", "train/boxes/cls", "train/boxes/dfl"} <= logged
    assert {"val/boxes/map/map", "val/boxes/map/map_50", "val/boxes/map/map_75"} <= logged


@pytest.mark.e2e
@pytest.mark.slow
def test_the_classes_the_descriptor_names_size_the_head(tmp_path: Path) -> None:
    """No config restates the class count: the descriptor declares it, the profile records
    it, and the network is rebuilt at that width — the same derived channel every head is
    sized through.
    """
    descriptor = write_yolo_dataset(tmp_path)
    config = load_config(
        {
            "image_size": [SIZE, SIZE],
            "data": {"source": descriptor, "inputs": {}},
            "tasks": {"boxes": {"preset": "detection"}},
            "model": {"name": "yolo", "model_name": "yolov8n.yaml"},
            "loader": {"batch_size": 2},
            "trainer": {"max_epochs": 1, "accelerator": "cpu", "logger": False, "default_root_dir": str(tmp_path)},
            "run": {"directory": str(tmp_path / "run"), "train": False, "test": False},
        }
    )

    experiment = assemble(config)

    model = experiment.module.model
    assert isinstance(model, YoloModel)
    # The detection head's own class count — the width the descriptor's names decided.
    assert cast("Any", model.detector).model[-1].nc == len(CLASSES)
