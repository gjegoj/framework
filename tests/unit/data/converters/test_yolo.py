"""The yolo converter: a labels tree into the canon — denormalised, validated aloud, stable."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.converters.yolo import convert


def yolo_tree(root: Path) -> Path:
    """Two 20x10 train images (one a negative) and one val image, with a descriptor."""
    for stage, names in (("train", ("a", "b")), ("val", ("c",))):
        (root / "images" / stage).mkdir(parents=True)
        (root / "labels" / stage).mkdir(parents=True)
        for name in names:
            cv2.imwrite(str(root / "images" / stage / f"{name}.jpg"), np.zeros((10, 20, 3), dtype=np.uint8))
    # Normalised cxcywh: centre (0.5, 0.5), size (0.5, 0.8) of a 20x10 image is [5, 1, 15, 9].
    (root / "labels" / "train" / "a.txt").write_text("1 0.5 0.5 0.5 0.8\n")
    (root / "labels" / "train" / "b.txt").write_text("")
    (root / "labels" / "val" / "c.txt").write_text("0 0.5 0.5 1.0 1.0\n")
    descriptor = root / "data.yaml"
    descriptor.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n")
    return descriptor


def test_the_tree_converts_to_one_sorted_canon_file_per_stage(tmp_path: Path) -> None:
    """Pixels, names and one row per image — what the boxes encoder reads without a flag."""
    convert(yolo_tree(tmp_path), into=tmp_path / "canon")

    assert (tmp_path / "canon" / "train.jsonl").read_text() == (
        '{"image": "images/train/a.jpg", "objects": [{"box": [5.0, 1.0, 15.0, 9.0], "class": "dog"}]}\n'
        '{"image": "images/train/b.jpg", "objects": []}\n'
    )
    assert '"class": "cat"' in (tmp_path / "canon" / "val.jsonl").read_text()


def test_a_stage_the_descriptor_never_declares_gets_no_file(tmp_path: Path) -> None:
    """A YOLO dataset without a test split simply has none; inventing one would be a lie."""
    convert(yolo_tree(tmp_path), into=tmp_path / "canon")

    assert not (tmp_path / "canon" / "test.jsonl").exists()


def test_rerunning_produces_a_byte_identical_file(tmp_path: Path) -> None:
    """Sorted rows and rounded corners are what make a canon file diff cleanly in review."""
    descriptor = yolo_tree(tmp_path)

    convert(descriptor, into=tmp_path / "canon")
    first = (tmp_path / "canon" / "train.jsonl").read_bytes()
    convert(descriptor, into=tmp_path / "canon")

    assert (tmp_path / "canon" / "train.jsonl").read_bytes() == first


def test_an_out_of_bounds_box_is_clipped_and_counted(tmp_path: Path) -> None:
    """Silent clipping is how a dataset quietly teaches boxes that leave the picture."""
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("1 0.5 0.5 1.2 0.8\n")

    report = convert(descriptor, into=tmp_path / "canon")

    assert report.clipped == 1
    assert report.clipped_files == {"images/train/a.jpg"}


def test_a_box_with_no_area_left_is_refused_by_the_image_it_came_from(tmp_path: Path) -> None:
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("1 0.0 0.0 0.0 0.0\n")

    with pytest.raises(ValueError, match="a.jpg"):
        convert(descriptor, into=tmp_path / "canon")


def test_an_annotated_image_that_is_missing_refuses_by_path(tmp_path: Path) -> None:
    """The labels say the image exists; skipping it would drop annotations in silence."""
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "images" / "train" / "a.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="images/train/a.jpg"):
        convert(descriptor, into=tmp_path / "canon")


def test_the_report_counts_what_it_wrote(tmp_path: Path) -> None:
    report = convert(yolo_tree(tmp_path), into=tmp_path / "canon")

    assert (report.images, report.objects) == (3, 2)


def test_a_dataset_under_a_parent_images_directory_keeps_its_annotations(tmp_path: Path) -> None:
    """Only the *last* ``images`` segment names the convention; rewriting every one sent
    the label lookup to a directory that does not exist and every row went negative."""
    descriptor = yolo_tree(tmp_path / "images" / "pets")

    report = convert(descriptor, into=tmp_path / "canon")

    assert report.objects == 2


def test_a_relative_path_in_the_descriptor_resolves_against_the_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reference tool resolves ``path:`` against the yaml; resolving against the
    process's directory converted zero images from any other CWD — silently."""
    descriptor = yolo_tree(tmp_path / "ds")
    descriptor.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n", encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = convert(descriptor, into=tmp_path / "canon")

    assert report.images == 3


def test_detritus_beside_the_pictures_is_not_an_image(tmp_path: Path) -> None:
    """.DS_Store, ultralytics' own *.cache, a classes.txt — none is a lost annotation.
    Opening them as pictures raised ``UnidentifiedImageError`` naming nothing."""
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "images" / "train" / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "images" / "train" / "train.cache").write_bytes(b"\x00")

    report = convert(descriptor, into=tmp_path / "canon")

    assert report.images == 3


def test_a_polygon_label_line_is_refused_naming_the_file_the_line_and_the_form(tmp_path: Path) -> None:
    """A YOLO-seg export in a detection tree is a common mistake; 'too many values to
    unpack' names none of the three things a user needs to find it."""
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("1 0.1 0.1 0.2 0.1 0.3 0.2 0.1 0.4\n")

    with pytest.raises(ValueError, match=r"a\.txt:1.*5 normalised fields.*polygon"):
        convert(descriptor, into=tmp_path / "canon")


def test_a_label_line_that_is_not_numbers_is_refused_at_its_address(tmp_path: Path) -> None:
    """Five fields of the right count and the wrong kind: the address still comes first."""
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("dog 0.5 0.5 0.5 0.8\n")

    with pytest.raises(ValueError, match=r"a\.txt:1.*class cx cy w h"):
        convert(descriptor, into=tmp_path / "canon")


def test_a_class_index_the_descriptor_never_declares_lists_the_declared_ones(tmp_path: Path) -> None:
    descriptor = yolo_tree(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("7 0.5 0.5 0.5 0.8\n")

    with pytest.raises(ValueError, match=r"a\.txt:1.*0: cat"):
        convert(descriptor, into=tmp_path / "canon")


def test_a_declared_stage_with_no_images_refuses_naming_key_and_directory(tmp_path: Path) -> None:
    """A declared stage is a promise; zero images against it is a resolution mistake
    (a wrong ``path:``, a moved directory), and an empty canon file written under it
    reads as a converted dataset."""
    descriptor = yolo_tree(tmp_path)
    descriptor.write_text(
        descriptor.read_text(encoding="utf-8").replace("images/val", "images/absent"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="images/absent"):
        convert(descriptor, into=tmp_path / "canon")

    assert not (tmp_path / "canon" / "val.jsonl").exists()
