"""A RAM cache: arrays decoded once in the parent, read by every worker from shared memory, never copied per process."""

from __future__ import annotations

import multiprocessing as mp
import pickle
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from src.core import Sample, require_tensor
from src.data import StandardPreprocessor
from src.data.cache import BYTES_PER_GIB, Cache, Key, RamCache
from src.data.encoders import ImageEncoder, MaskEncoder, MultilabelEncoder
from src.data.preprocessor import BATCH_PER_WORKER
from src.data.registry import cache_registry
from src.transforms import SampleTransform
from tests.support.declarations import CLASSES
from tests.unit.data.conftest import PreprocessorFactory, pipeline

KEY = ("inputs", "image", "a.png")

CHILD_PATIENCE = 60
"""How long the parent waits for a spawned child, at every point it could wait forever instead."""

PICTURE = (6, 8, 3)
"""What every picture written by `pictures` decodes to; a budget is sized against it."""


def pictures(root: Path, count: int) -> list[str]:
    """``count`` distinct ONGs under ``root``, named by number."""
    for i in range(count):
        cv2.imwrite(str(root / f"{i}.png"), np.full(PICTURE, i, dtype=np.uint8))
    return [f"{i}.png" for i in range(count)]


def room_for_one_picture() -> float:
    """A budget, in GiB, that holds one decoded picture and not two."""
    return 1.5 * int(np.prod(PICTURE)) / BYTES_PER_GIB


@pytest.fixture
def cache() -> RamCache:
    return RamCache(max_gib=0.001, workers=2)  # ~1 MiB


def stored(cache: Cache, key: Key) -> np.ndarray:
    """The array a test expects to find; a miss fails here, by key."""
    value = cache.get(key)
    assert value is not None, key
    return value


def test_every_registered_cache_serves_the_contract() -> None:
    assert "ram" in cache_registry


class TestArena:
    def test_round_trips_an_array_and_hands_back_a_copy(self, cache: RamCache) -> None:
        image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        with cache.filling():
            assert cache.put(KEY, image)

        hit = stored(cache, KEY)
        assert np.array_equal(hit, image) and hit.dtype == image.dtype
        hit[0, 0, 0] = 99
        assert stored(cache, KEY)[0, 0, 0] == 0

    def test_keys_are_namespaced_so_one_path_may_hold_a_image_and_a_mask(self, cache: RamCache) -> None:
        with cache.filling():
            cache.put(("inputs", "image", "a.png"), np.ones((2, 2, 3), np.uint8))
            cache.put(("targets", "mask", "a.png"), np.zeros((2, 2), np.int64))

        assert stored(cache, ("targets", "mask", "a.png")).dtype == np.int64
        assert cache.usage.files == 2

    def test_stores_only_arrays_and_only_while_filling(self, cache: RamCache) -> None:
        assert not cache.put(KEY, np.ones(3, np.uint8))  # outside a filling window
        with cache.filling():
            assert not cache.put(("t", "label", "x"), "cat")
        assert cache.get(KEY) is None and cache.usage.files == 0

    def test_a_full_budget_declines_the_rest_and_says_so(self, cache: RamCache) -> None:
        big = np.zeros(600_000, np.uint8)
        with cache.filling():
            assert cache.put(("a",), big) and not cache.put(("b",), big)

        assert cache.usage.full and cache.usage.declined == 1 and cache.get(("b",)) is None
        assert "1" in cache.summary()

    def test_allocations_never_exceed_the_budget_and_grow_by_need(self) -> None:
        cache = RamCache(max_gib=0.002, workers=1)  # ~2 MiB budget, below one nominal segment
        with cache.filling():
            cache.put(("a",), np.zeros(1_000_000, np.uint8))

        usage = cache.usage
        assert usage.allocated_bytes <= usage.capacity_bytes and usage.used_bytes <= usage.allocated_bytes
        assert len(cache.segments) == 1 and cache.segments[0].numel() == usage.capacity_bytes

    def test_membership_does_not_copy_the_array(self, cache: RamCache) -> None:
        with cache.filling():
            cache.put(KEY, np.ones(3, np.uint8))

        assert KEY in cache and ("x",) not in cache

    def test_a_cache_that_reached_another_process_refuses_to_be_filled(self, cache: RamCache) -> None:
        """Two fillers over one shared arena overwrite each other's bytes — pixels quietly wrong, not a
        crash — and `ddp_spawn` runs setup in every rank, so the copy refuses rather than trusting."""
        with cache.filling():
            cache.put(KEY, np.ones(3, np.uint8))
        clone = pickle.loads(ForkingPickler.dumps(cache))

        assert clone.put(("elsewhere",), np.ones(3, np.uint8)) is False
        with pytest.raises(RuntimeError, match="parent"), clone.filling():
            pass
        assert clone.usage.files == cache.usage.files

    def test_the_worker_pickler_sends_shared_handles_not_bytes(self, cache: RamCache) -> None:
        with cache.filling():
            cache.put(KEY, np.full((4, 4), 7, np.uint8))

        clone = pickle.loads(ForkingPickler.dumps(cache))

        original, copied = cache.get(KEY), clone.get(KEY)
        assert clone.segments[0].untyped_storage().is_shared()
        assert original is not None and copied is not None and np.array_equal(copied, original)

    def test_a_spawned_process_writes_into_the_arena_the_parent_reads(self, cache: RamCache) -> None:
        """Real sharing, not equal copies: the child pokes a byte and the parent sees it.

        Every wait is bounded. Read straight off the pipe, a child that died before writing — shared
        memory refused by a sandbox is how this turned up — leaves the parent blocked forever, and the
        join that was meant to time out is never reached: the whole gate hangs with nothing said.
        """
        with cache.filling():
            cache.put(KEY, np.arange(16, dtype=np.uint8))
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_poke_in_child, args=(cache, child))
        process.start()
        try:
            # The parent's copy of the writing end, so a child that dies without writing reads as an
            # end of file rather than as a message still to come.
            child.close()
            assert parent.poll(CHILD_PATIENCE), f"the child wrote nothing within {CHILD_PATIENCE}s"
            received = parent.recv()
            process.join(timeout=CHILD_PATIENCE)
            assert process.exitcode == 0, f"the child exited with {process.exitcode}"
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=CHILD_PATIENCE)

        assert received == list(range(16))
        assert stored(cache, KEY)[0] == 42

    def test_a_data_loader_with_spawned_workers_reads_cached_samples(
        self, make_preprocessor: PreprocessorFactory
    ) -> None:
        preprocessor = make_preprocessor(targets={}, cache=RamCache(max_gib=0.01, workers=2))
        rows = [Sample(inputs={"image": f"{name}.png"}) for name in "abc"]
        preprocessor.warm(rows, label="test")
        loader = DataLoader(
            Rows(rows, preprocessor, pipeline(preprocessor)),
            batch_size=3,
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=preprocessor.collate,
        )

        batch = next(iter(loader))

        assert len(batch) == 3 and require_tensor(batch.inputs["image"], name="image").shape == (3, 3, 4, 4)

    @pytest.mark.parametrize("kwargs", [{"max_gib": 0}, {"workers": 0}], ids=["no budget", "no workers"])
    def test_refuses_a_useless_declaration(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RamCache(**kwargs)


def _poke_in_child(cache: RamCache, pipe: Connection) -> None:
    value = cache.get(KEY)
    cache.segments[0][0] = 42
    pipe.send(None if value is None else value.tolist())


class Rows(Dataset[Sample]):
    def __init__(self, rows: list[Sample], preprocessor: StandardPreprocessor, transform: SampleTransform) -> None:
        self.rows, self.preprocessor, self.transform = rows, preprocessor, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        return self.preprocessor.preprocess(self.rows[index], self.transform)


class TestWarm:
    @pytest.fixture
    def counting(self, images: Path) -> type[CountingImage]:
        CountingImage.loads = 0
        return CountingImage

    @pytest.fixture
    def preprocessor(
        self,
        make_preprocessor: PreprocessorFactory,
        counting: type[CountingImage],
        mask_encoder: MaskEncoder,
        images: Path,
        cache: RamCache,
    ) -> StandardPreprocessor:
        return make_preprocessor(
            inputs={"image": counting(image_size=(4, 4), root=images)}, targets={"mask": mask_encoder}, cache=cache
        )

    def test_warming_decodes_every_file_once_and_serves_it_from_memory_afterwards(
        self, preprocessor: StandardPreprocessor, counting: type[CountingImage], cache: RamCache
    ) -> None:
        rows = [Sample(inputs={"image": f"{name}.png"}, targets={"mask": f"{name}_mask.png"}) for name in "aab"]

        preprocessor.warm(rows, label="train")
        counting.loads = 0
        prepare = pipeline(preprocessor)
        for row in rows:
            preprocessor.preprocess(row, prepare)

        assert cache.usage.files == 4  # two images + two masks
        assert counting.loads == 0

    def test_cells_that_are_not_files_never_reach_the_cache_even_as_lists(
        self, make_preprocessor: PreprocessorFactory, cache: RamCache
    ) -> None:
        preprocessor = make_preprocessor(targets={"tags": MultilabelEncoder(classes=CLASSES)}, cache=cache)

        preprocessor.warm(iter([Sample(inputs={"image": "a.png"}, targets={"tags": ["cat", "dog"]})]), label="train")

        assert cache.usage.files == 1

    def test_an_unreadable_file_is_skipped_not_fatal(self, preprocessor: StandardPreprocessor, cache: RamCache) -> None:
        preprocessor.warm([Sample(inputs={"image": "missing.png"})], label="val")

        assert cache.usage.files == 0

    def test_a_full_budget_ends_the_reading_and_not_only_the_storing(
        self,
        tmp_path: Path,
        make_preprocessor: PreprocessorFactory,
        counting: type[CountingImage],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every file after the one that did not fit would be decoded only to be declined — the cost warming
        exists to remove. The batch already in flight is finished, and the line says where reading stopped."""
        names = pictures(tmp_path, count=3 * BATCH_PER_WORKER)
        cache = RamCache(max_gib=room_for_one_picture(), workers=1)
        preprocessor = make_preprocessor(
            inputs={"image": counting(image_size=(4, 4), root=tmp_path)}, targets={}, cache=cache
        )

        with caplog.at_level("INFO"):
            preprocessor.warm([Sample(inputs={"image": name}) for name in names], label="train")

        assert cache.usage.full and cache.usage.files == 1
        assert counting.loads == cache.workers * BATCH_PER_WORKER
        assert any(f"{counting.loads} of {len(names)}" in one.message for one in caplog.records)

    def test_a_split_the_budget_filled_before_is_not_walked_and_says_so(
        self,
        tmp_path: Path,
        make_preprocessor: PreprocessorFactory,
        counting: type[CountingImage],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Its rows would be built from the table one by one, for a cache that declines every one of them;
        and a skip nobody hears of looks like a warmed cache that is not."""
        names = pictures(tmp_path, count=2)
        cache = RamCache(max_gib=room_for_one_picture(), workers=1)
        preprocessor = make_preprocessor(
            inputs={"image": counting(image_size=(4, 4), root=tmp_path)}, targets={}, cache=cache
        )
        preprocessor.warm([Sample(inputs={"image": name}) for name in names], label="train")
        later = iter([Sample(inputs={"image": names[1]})])

        with caplog.at_level("INFO"):
            preprocessor.warm(later, label="val")

        assert next(later, None) is not None
        assert any("val" in one.message and "not cached" in one.message for one in caplog.records)


class CountingImage(ImageEncoder):
    loads: ClassVar[int] = 0

    def load(self, value: object) -> np.ndarray:
        type(self).loads += 1
        return super().load(value)


def test_the_budget_is_the_machines_and_is_divided_between_the_copies_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several devices means several copies of this script, each with an arena the others cannot read:
    read per process, `max_gib: 8` would quietly mean 64 GiB on an eight-GPU node."""
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    alone = RamCache(max_gib=1.0).capacity

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    assert RamCache(max_gib=1.0).capacity == alone // 4
