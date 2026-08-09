"""One cache, many columns: a shared budget must not mean a shared key.

The standard segmentation layout has an image and a mask under the same
filename in different folders. Keying on the cell value alone made the second
read of that name serve the first one's array.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.data.cache import RamCache, cached


def loader_returning(value: np.ndarray) -> Any:
    def load(_: Any) -> np.ndarray:
        return value

    return load


def counting_cache() -> tuple[RamCache, dict[str, int], Callable[[Any], np.ndarray]]:
    """A cache, a read counter, and the loader that increments it.

    Every test here asks the same question — how many files were actually read — so the
    three travel together rather than being rebuilt with the counter wired by hand.
    """
    cache = RamCache(max_gib=1.0)
    counted = {"reads": 0}

    def load(_: Any) -> np.ndarray:
        counted["reads"] += 1
        return np.zeros((2, 2), dtype=np.uint8)

    return cache, counted, load


def test_two_columns_with_one_filename_keep_their_own_arrays() -> None:
    """The failure this prevents is silent: shapes may match and training just learns the wrong thing."""
    cache = RamCache(max_gib=1.0)
    photo, mask = np.ones((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.int64)
    read_photo = cached(loader_returning(photo), cache.scoped("image"))
    read_mask = cached(loader_returning(mask), cache.scoped("mask"))

    # Warmed through the plain cache, as the data module does: scoping lives only in
    # the wrapped loaders, applied where they were built.
    cache.warm(["frame.png"], read_photo)
    cache.warm(["frame.png"], read_mask)

    assert read_photo("frame.png").shape == photo.shape
    assert read_mask("frame.png").shape == mask.shape


def test_warming_one_column_does_not_mark_the_other_as_done() -> None:
    """Skipping a key already held is what made the mask read zero files."""
    cache, counted, load = counting_cache()

    cache.warm(["a.png"], cached(load, cache.scoped("image")))
    cache.warm(["a.png"], cached(load, cache.scoped("mask")))

    assert counted["reads"] == 2


def test_one_column_still_reads_each_file_once() -> None:
    """The namespace must not cost the dedup that makes warming worth doing."""
    cache, counted, load = counting_cache()

    cache.warm(["a.png", "a.png", "b.png"], cached(load, cache.scoped("image")))

    assert counted["reads"] == 2


def test_a_second_epoch_of_warming_reads_nothing() -> None:
    """Warming is idempotent, which is what lets train and val be warmed in turn."""
    cache, counted, load = counting_cache()

    reader = cached(load, cache.scoped("image"))
    cache.warm(["a.png"], reader)
    cache.warm(["a.png"], reader)

    assert counted["reads"] == 1
