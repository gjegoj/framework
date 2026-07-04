"""The in-RAM image/mask cache: array store, caching decorators, DataModule warm-up."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CacheOptions,
    CsvDataSource,
    DataModule,
    LabelEncoder,
    TargetBinding,
)
from tests.support.builders import make_transform as _make_transform

_LABEL_MAPPING: dict[int, str] = {0: "cat", 1: "cow", 2: "dog"}

LABELS = ["cat", "dog", "cow"]


@pytest.fixture
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 RGB jpgs across 3 classes and a CSV indexing them."""
    return make_image_csv(count=15, size=32, seed=0, labels=LABELS)


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping=_LABEL_MAPPING))


class TestArrayCache:
    def test_warm_then_get_hit_and_miss(self) -> None:
        from src.data.cache import ArrayCache

        store = {"a": np.zeros((2, 2), np.uint8), "b": np.ones((2, 2), np.uint8)}
        cache = ArrayCache(max_bytes=10**6)
        cache.warm(["a", "b"], store.__getitem__, workers=2)
        assert cache.get("a") is not None
        assert cache.get("missing") is None
        assert cache.nbytes == store["a"].nbytes + store["b"].nbytes

    def test_byte_cap_stops_filling(self) -> None:
        from src.data.cache import ArrayCache

        items = {str(i): np.zeros((100, 100), np.uint8) for i in range(10)}  # 10 KB each
        cache = ArrayCache(max_bytes=25_000)  # room for ~2 items
        cache.warm(list(items), items.__getitem__, workers=1)
        assert cache.nbytes <= 25_000
        assert 0 < sum(cache.get(k) is not None for k in items) < 10

    def test_bad_item_is_skipped_not_fatal(self) -> None:
        from src.data.cache import ArrayCache

        def load(key: str) -> np.ndarray:
            if key == "bad":
                raise FileNotFoundError(key)
            return np.zeros((2, 2), np.uint8)

        cache = ArrayCache(max_bytes=10**6)
        cache.warm(["ok", "bad"], load, workers=2)
        assert cache.get("ok") is not None
        assert cache.get("bad") is None

    def test_disabled_cache_caches_nothing(self) -> None:
        from src.data.cache import ArrayCache

        cache = ArrayCache(max_bytes=0)
        cache.warm(["a"], lambda _: np.zeros((2, 2), np.uint8), workers=1)
        assert cache.get("a") is None
        assert cache.nbytes == 0


class TestCachingDecorators:
    def test_caching_loader_hit_and_miss(self) -> None:
        from src.data.cache import ArrayCache, CachingLoader
        from src.data.loaders import ImageLoader

        cache = ArrayCache(max_bytes=10**6)
        cached = np.full((2, 2, 3), 7, np.uint8)
        cache.warm(["/hit.png"], lambda _: cached, workers=1)
        loader = CachingLoader(ImageLoader(), cache)
        assert loader.file_based is True
        assert np.array_equal(loader.load("/hit.png"), cached)  # served from cache
        with pytest.raises(FileNotFoundError):
            loader.load("/does-not-exist.png")  # miss falls through to the inner loader

    def test_caching_codec_delegates_and_serves(self) -> None:
        from src.data.cache import ArrayCache, CachingTargetEncoder
        from src.data.encoders import MaskEncoder

        cache = ArrayCache(max_bytes=10**6)
        cached = np.array([[1, 2], [3, 4]], np.uint8)
        cache.warm(["/m.png"], lambda _: cached, workers=1)
        codec = CachingTargetEncoder(MaskEncoder(class_mapping={0: "bg", 1: "fg"}), cache)
        assert codec.file_based is True  # both file flags delegated to the inner encoder
        assert codec.spatial is True
        assert codec.num_classes == 2  # delegated
        assert np.array_equal(codec.load("/m.png"), cached)  # served from cache
        assert codec.to_tensor(cached).dtype.is_floating_point is False  # delegated (long)


class TestDataModuleCache:
    def test_cache_warms_wraps_and_serves(self, csv_path: Path) -> None:
        from src.data.cache import CachingLoader

        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            cache_options=CacheOptions(max_bytes=10**9),
        )
        dm.setup()
        assert isinstance(dm._input_bindings[0].loader, CachingLoader)
        assert dm._cache is not None and dm._cache.nbytes > 0
        first_path = str(dm._datasets[Stage.TRAIN][0]._frame.iloc[0]["image_path"])
        assert dm._cache.get(first_path) is not None  # warmed (no root_path → key == path)
        sample = dm._datasets[Stage.TRAIN][0][0]  # end-to-end getitem still works (first source, first sample)
        assert "image" in sample.inputs

    def test_cache_disabled_leaves_plain_loader(self, csv_path: Path) -> None:
        from src.data.loaders import ImageLoader

        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
        )
        dm.setup()
        assert isinstance(dm._input_bindings[0].loader, ImageLoader)
        assert dm._cache is None
