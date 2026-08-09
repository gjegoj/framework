"""``ImageLoader``: image files into the HWC RGB uint8 arrays transforms expect."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import ImageLoader
from src.data.registry import input_loader_registry

BLUE_IN_BGR = (255, 0, 0)


def write_image(path: Path, color: tuple[int, int, int] = BLUE_IN_BGR, size: int = 4) -> Path:
    """Write a solid-colour PNG; ``color`` is in OpenCV's BGR order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size, size, 3), color, dtype=np.uint8))
    return path


def test_reads_an_image_as_hwc_uint8(tmp_path: Path) -> None:
    image = ImageLoader()(write_image(tmp_path / "picture.png"))

    assert isinstance(image, np.ndarray)
    assert image.shape == (4, 4, 3)
    assert image.dtype == np.uint8


def test_channels_come_back_in_rgb_order(tmp_path: Path) -> None:
    """A blue image must read as RGB (0, 0, 255) — not OpenCV's native BGR."""
    image = ImageLoader()(write_image(tmp_path / "blue.png", color=BLUE_IN_BGR))

    assert tuple(image[0, 0]) == (0, 0, 255)


def test_root_is_prepended_to_table_paths(tmp_path: Path) -> None:
    write_image(tmp_path / "images" / "one.png")

    image = ImageLoader(root=tmp_path / "images")("one.png")

    assert image.shape == (4, 4, 3)


def test_without_a_root_paths_are_used_as_given(tmp_path: Path) -> None:
    path = write_image(tmp_path / "one.png")

    assert ImageLoader()(str(path)).shape == (4, 4, 3)


def test_grayscale_returns_a_single_plane(tmp_path: Path) -> None:
    image = ImageLoader(grayscale=True)(write_image(tmp_path / "picture.png"))

    assert image.shape == (4, 4)
    assert image.dtype == np.uint8


def test_a_missing_file_names_the_resolved_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.png"):
        ImageLoader(root=tmp_path)("absent.png")


def test_an_undecodable_file_is_reported_as_such(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not really an image")

    with pytest.raises(ValueError, match="decode"):
        ImageLoader()(broken)


def test_registered_under_the_image_key() -> None:
    assert set(input_loader_registry) == {"image"}
    assert isinstance(input_loader_registry.create("image"), ImageLoader)
