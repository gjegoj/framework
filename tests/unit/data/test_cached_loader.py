"""The one write path into a cache, and the encoder that reaches it from inside."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data import ImageLoader, MaskTargetEncoder, RamCache, cached


def write_mask(root: Path, name: str = "m.png") -> str:
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / name), np.zeros((8, 8), dtype=np.uint8))
    return name


def test_a_hit_is_served_without_reading_again() -> None:
    reads: list[str] = []
    cache = RamCache(max_gib=1.0)

    def load(value: Any) -> np.ndarray:
        reads.append(str(value))
        return np.zeros(4, dtype=np.uint8)

    reader = cached(load, cache)
    cache.warm(["a.png"], reader)
    first = reader("a.png")
    second = reader("a.png")

    assert reads == ["a.png"]
    assert first is second


def test_a_miss_falls_through_to_the_loader() -> None:
    cache = RamCache(max_gib=1.0)
    reader = cached(lambda value: np.full(4, 7, dtype=np.uint8), cache)

    assert reader("never-warmed.png").tolist() == [7, 7, 7, 7]


def test_reading_outside_warm_up_does_not_fill_the_cache() -> None:
    """Otherwise every DataLoader worker would grow a private copy after the fork."""
    cache = RamCache(max_gib=1.0)
    reader = cached(lambda value: np.zeros(4, dtype=np.uint8), cache)

    reader("a.png")

    assert cache.get("a.png") is None


def test_a_mask_encoder_reads_through_the_cache(tmp_path: Path) -> None:
    """Warmed by ``load`` — the reading half, and the same call the dataset makes."""
    cache = RamCache(max_gib=1.0)
    name = write_mask(tmp_path)
    encoder = MaskTargetEncoder(num_classes=2, root=tmp_path, cache=cache)

    cache.warm([name], encoder.load)

    assert cache.get(name) is not None


def test_a_mask_encoder_without_a_cache_behaves_as_before(tmp_path: Path) -> None:
    name = write_mask(tmp_path)
    encoder = MaskTargetEncoder(num_classes=2, root=tmp_path)

    assert encoder.load(name).shape == (8, 8)


def test_an_uncached_loader_is_untouched(tmp_path: Path) -> None:
    """Wrapping is a decision made at assembly; the loader itself knows nothing about it."""
    name = write_mask(tmp_path)
    loader = ImageLoader(root=tmp_path, grayscale=True)

    assert loader(name).shape == (8, 8)
