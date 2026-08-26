"""Per-pixel targets read from mask files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.encoders import MaskTargetEncoder


def write_mask(path: Path, classes: np.ndarray) -> Path:
    """Write a class-index mask as a grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), classes.astype(np.uint8))
    return path


def test_mask_encoder_reads_class_indices_as_an_integer_array(tmp_path: Path) -> None:
    """Reading is ``load``, the half that runs before the transforms — a mask has to be
    pixels by then, or the pipeline's geometry has nothing to move."""
    classes = np.array([[0, 1], [2, 0]])
    path = write_mask(tmp_path / "mask.png", classes)

    loaded = MaskTargetEncoder(num_classes=3).load(path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (2, 2)
    assert loaded.dtype == np.int64
    assert loaded.tolist() == classes.tolist()


def test_mask_encoder_encodes_the_pixels_the_transform_returned(tmp_path: Path) -> None:
    """``encode`` runs after the transforms, so it is handed pixels, not a path."""
    moved = np.array([[1, 0], [0, 2]], dtype=np.int64)

    assert MaskTargetEncoder(num_classes=3).encode(moved).tolist() == moved.tolist()


def test_mask_encoder_prepends_its_own_root(tmp_path: Path) -> None:
    write_mask(tmp_path / "masks" / "one.png", np.zeros((2, 2)))

    loaded = MaskTargetEncoder(num_classes=2, root=tmp_path / "masks").load("one.png")

    assert loaded.shape == (2, 2)


def test_mask_encoder_reports_the_class_count_it_was_given() -> None:
    assert MaskTargetEncoder(num_classes=4).num_classes == 4


def test_mask_encoder_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.png"):
        MaskTargetEncoder(num_classes=2, root=tmp_path).load("absent.png")


def test_a_mask_derives_its_count_from_declared_classes() -> None:
    encoder = MaskTargetEncoder(classes={0: "background", 1: "defect"})

    assert encoder.num_classes == 2
    assert encoder.class_names == ["background", "defect"]


def test_a_mask_refuses_disagreeing_count_and_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        MaskTargetEncoder(num_classes=3, classes={0: "background", 1: "defect"})


def test_a_mask_needs_a_count_from_somewhere() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        MaskTargetEncoder()
