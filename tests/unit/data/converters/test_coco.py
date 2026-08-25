"""The COCO converter: category ids to names, xywh to xyxy, crowd objects dropped out loud."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.data.converters.coco import convert


def coco_export(root: Path, annotations: list[dict[str, Any]] | None = None) -> Path:
    """Two 20x10 images; by default one carries a dog and a crowd, the other nothing."""
    (root / "images").mkdir()
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(root / "images" / name), np.zeros((10, 20, 3), dtype=np.uint8))
    export = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 20, "height": 10},
            {"id": 2, "file_name": "b.jpg", "width": 20, "height": 10},
        ],
        "categories": [{"id": 7, "name": "dog"}, {"id": 9, "name": "cat"}],
        "annotations": (
            annotations
            if annotations is not None
            else [
                {"image_id": 1, "category_id": 7, "bbox": [5.0, 1.0, 10.0, 8.0], "iscrowd": 0},
                {"image_id": 1, "category_id": 9, "bbox": [0.0, 0.0, 4.0, 4.0], "iscrowd": 1},
            ]
        ),
    }
    path = root / "instances.json"
    path.write_text(json.dumps(export))
    return path


def test_the_export_converts_to_names_and_pixel_corners(tmp_path: Path) -> None:
    """COCO's ``[x, y, w, h]`` is the same box as the canon's ``[x1, y1, x2, y2]``, said once."""
    convert(coco_export(tmp_path), images=tmp_path / "images", into=tmp_path / "canon")

    rows = [json.loads(line) for line in (tmp_path / "canon" / "annotations.jsonl").read_text().splitlines()]

    assert rows[0]["image"] == "a.jpg"
    assert rows[0]["objects"] == [{"box": [5.0, 1.0, 15.0, 9.0], "class": "dog"}]


def test_an_image_with_no_kept_annotations_still_gets_its_row(tmp_path: Path) -> None:
    """A negative is an observation; dropping the row would drop the picture from the run."""
    convert(coco_export(tmp_path), images=tmp_path / "images", into=tmp_path / "canon")

    rows = [json.loads(line) for line in (tmp_path / "canon" / "annotations.jsonl").read_text().splitlines()]

    assert rows[1] == {"image": "b.jpg", "objects": []}


def test_crowd_objects_are_dropped_and_counted(tmp_path: Path) -> None:
    """The standard detection stance — but said aloud, because a silent drop reads as clean data."""
    report = convert(coco_export(tmp_path), images=tmp_path / "images", into=tmp_path / "canon")

    assert report.dropped_crowd == 1
    assert report.objects == 1


def test_a_box_leaving_its_image_is_clipped_and_counted(tmp_path: Path) -> None:
    export = coco_export(
        tmp_path,
        annotations=[{"image_id": 1, "category_id": 7, "bbox": [5.0, 1.0, 100.0, 8.0], "iscrowd": 0}],
    )

    report = convert(export, images=tmp_path / "images", into=tmp_path / "canon")

    assert report.clipped == 1
    rows = [json.loads(line) for line in (tmp_path / "canon" / "annotations.jsonl").read_text().splitlines()]
    assert rows[0]["objects"] == [{"box": [5.0, 1.0, 20.0, 9.0], "class": "dog"}]


def test_an_image_the_export_names_but_the_directory_lacks_refuses_by_path(tmp_path: Path) -> None:
    export = coco_export(tmp_path)
    (tmp_path / "images" / "a.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="a.jpg"):
        convert(export, images=tmp_path / "images", into=tmp_path / "canon")


def test_an_annotation_whose_image_the_export_does_not_describe_is_refused(tmp_path: Path) -> None:
    """An orphaned image_id is a corrupt export, not a stance: dropping crowds is
    documented and counted, but nobody decided to drop this one."""
    export = coco_export(
        tmp_path,
        annotations=[
            {"image_id": 1, "category_id": 7, "bbox": [5.0, 1.0, 10.0, 8.0], "iscrowd": 0},
            {"image_id": 99, "category_id": 7, "bbox": [1.0, 1.0, 2.0, 2.0], "iscrowd": 0},
        ],
    )

    with pytest.raises(ValueError, match="99"):
        convert(export, images=tmp_path / "images", into=tmp_path / "canon")
