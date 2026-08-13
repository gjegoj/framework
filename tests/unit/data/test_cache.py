"""``RamCache``: what it keeps, what it refuses, and when it stops accepting."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.data.cache import BYTES_PER_GIB, RamCache
from src.data.registry import cache_registry


def array(megabytes: float = 1.0) -> np.ndarray:
    return np.zeros(int(megabytes * 1024 * 1024), dtype=np.uint8)


def warmed(cache: RamCache, values: dict[str, object]) -> RamCache:
    """Warm ``cache`` with a loader that serves ``values``, the way a real one would."""

    def load(key: str) -> object:
        result = values[key]
        cache.put(key, result)
        return result

    cache.warm(list(values), load)
    return cache


def test_a_warmed_key_is_served_from_memory() -> None:
    stored = array()
    cache = warmed(RamCache(max_gib=1.0), {"a.png": stored})

    assert cache.get("a.png") is stored


def test_an_unknown_key_is_a_miss() -> None:
    assert RamCache(max_gib=1.0).get("nothing.png") is None


def test_only_arrays_are_kept() -> None:
    """A text loader returns a string; the type decides, so no component needs a flag."""
    cache = warmed(RamCache(max_gib=1.0), {"caption.txt": "a red car"})

    assert cache.get("caption.txt") is None


def test_the_budget_stops_the_cache_rather_than_the_run() -> None:
    values: dict[str, object] = {f"{index}.png": array(1.0) for index in range(20)}
    cache = warmed(RamCache(max_gib=5 / 1024), values)

    kept = sum(1 for key in values if cache.get(key) is not None)
    assert 0 < kept < 20


def test_writes_are_refused_once_warming_is_over() -> None:
    """DataLoader workers fork after warm-up; a worker filling the cache would grow its own copy."""
    cache = RamCache(max_gib=1.0)

    cache.put("late.png", array())

    assert cache.get("late.png") is None


def test_a_file_that_cannot_be_read_is_skipped_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    cache = RamCache(max_gib=1.0)

    def load(key: str) -> None:
        if key == "broken.png":
            raise OSError("cannot decode")
        cache.put(key, array())

    with caplog.at_level(logging.WARNING):
        cache.warm(["good.png", "broken.png"], load)

    assert "broken.png" in caplog.text
    assert cache.get("good.png") is not None


def test_a_key_already_held_is_not_read_again() -> None:
    cache = RamCache(max_gib=1.0)
    reads: list[str] = []

    def load(key: str) -> None:
        reads.append(key)
        cache.put(key, array())

    cache.warm(["a.png"], load)
    cache.warm(["a.png"], load)

    assert reads == ["a.png"]


def test_duplicate_keys_are_read_once() -> None:
    """Rows sharing a file are why the cache is keyed by value rather than by row."""
    cache = RamCache(max_gib=1.0)
    reads: list[str] = []

    def load(key: str) -> None:
        reads.append(key)
        cache.put(key, array())

    cache.warm(["a.png", "a.png", "a.png"], load)

    assert reads == ["a.png"]


@pytest.mark.parametrize(("kwargs", "message"), [({"max_gib": 0}, "positive max_gib"), ({"workers": 0}, "one worker")])
def test_contradictory_settings_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RamCache(**kwargs)  # type: ignore[arg-type]


def test_the_cache_is_reachable_from_config_by_name() -> None:
    assert isinstance(cache_registry.create("ram", max_gib=0.5), RamCache)


def test_one_gib_is_what_it_says() -> None:
    assert BYTES_PER_GIB == 1024**3


def test_usage_reports_what_is_held_against_the_budget() -> None:
    cache = warmed(RamCache(max_gib=1.0), {"a.png": array(2.0), "b.png": array(3.0)})

    usage = cache.usage()

    assert usage.files == 2
    assert usage.used_bytes == 5 * 1024 * 1024
    assert usage.capacity_bytes == BYTES_PER_GIB
    assert usage.declined == 0


def test_declined_counts_only_what_the_budget_turned_away() -> None:
    """A scalar target is never cached by design — counting it would report a
    full cache over an empty one. Only the file the budget refused is in the count."""
    values: dict[str, object] = {
        "kept.png": array(1.0),
        "temperature": 5300.0,
        "too-big.png": array(10.0),
    }

    cache = warmed(RamCache(max_gib=1.5 / 1024), values)

    assert cache.usage().declined == 1
