"""Decoded files held in shared memory: filled once in the parent, read by every data worker.

The arena itself is never duplicated per worker; a read copies out only the one array it asked for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Final

import numpy as np
import torch
from torch import Tensor

from src.data.registry import cache_registry

BYTES_PER_GIB: Final = 1024**3
SEGMENT_BYTES: Final = 256 * 1024**2
"""Arenas grow one shared segment at a time, never past the budget: no upfront reservation, few handles."""

type Key = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheUsage:
    files: int
    used_bytes: int
    allocated_bytes: int
    capacity_bytes: int
    declined: int
    full: bool


class Cache(ABC):
    """Writes happen only inside ``filling()``, in the parent process, before data workers exist.

    Afterwards the cache is read-only everywhere — a worker that misses reads the file itself — and a
    copy that reached another process refuses to be filled at all.
    """

    workers: int

    @abstractmethod
    def __contains__(self, key: Key) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: Key) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, key: Key, value: object) -> bool:
        """Store an array while filling; anything else, or outside the window, is declined quietly."""
        raise NotImplementedError

    @abstractmethod
    def filling(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def usage(self) -> CacheUsage:
        raise NotImplementedError

    def status(self) -> str:
        usage = self.usage
        return f"{usage.used_bytes / BYTES_PER_GIB:.2f}/{usage.capacity_bytes / BYTES_PER_GIB:.2f} GiB"

    def summary(self) -> str:
        usage = self.usage
        allocated = usage.allocated_bytes / BYTES_PER_GIB
        line = f"Cache holds {usage.files} file(s), {self.status()} ({allocated:.2f} GiB allocated)."
        if usage.full:
            line += f" Budget full: {usage.declined} file(s) declined; they and the rest read from disk each epoch."
        return line


@dataclass(frozen=True, slots=True)
class _Entry:
    segment: int
    offset: int
    shape: tuple[int, ...]
    dtype: str


@cache_registry.register("ram")
class RamCache(Cache):
    """Arrays laid end to end in shared-memory segments; the index travels to workers as plain data."""

    def __init__(self, max_gib: float = 4.0, workers: int = 8) -> None:
        if max_gib <= 0:
            raise ValueError(f"A ram cache needs a positive max_gib, got {max_gib}; omit the section to disable it.")
        if workers < 1:
            raise ValueError(f"A ram cache needs at least one worker, got {workers}.")
        self.workers = workers
        self.capacity = int(max_gib * BYTES_PER_GIB)
        self.segments: list[Tensor] = []
        self._index: dict[Key, _Entry] = {}
        self._used = 0
        self._allocated = 0
        self._cursor = 0
        self._declined = 0
        self._full = False
        self._filling = False
        self._may_fill = True
        self._lock = Lock()

    def __contains__(self, key: Key) -> bool:
        return key in self._index

    def get(self, key: Key) -> np.ndarray | None:
        entry = self._index.get(key)
        if entry is None:
            return None
        flat = self.segments[entry.segment][entry.offset : entry.offset + _nbytes(entry)].numpy()
        return flat.view(np.dtype(entry.dtype)).reshape(entry.shape).copy()

    def put(self, key: Key, value: object) -> bool:
        if not self._filling or not isinstance(value, np.ndarray) or value.dtype == object:
            return False
        array = np.ascontiguousarray(value)
        with self._lock:
            if key in self._index:
                return True
            if self._full or not self._room_for(array.nbytes):
                self._declined += 1
                self._full = True
                return False
            segment, offset = len(self.segments) - 1, self._cursor
            self.segments[segment][offset : offset + array.nbytes] = torch.from_numpy(array.view(np.uint8).reshape(-1))
            self._index[key] = _Entry(segment, offset, array.shape, array.dtype.str)
            self._cursor += array.nbytes
            self._used += array.nbytes
            return True

    def _room_for(self, nbytes: int) -> bool:
        """Grow by a segment sized to the remaining budget (allocations count, not just payload)."""
        if self.segments and self._cursor + nbytes <= self.segments[-1].numel():
            return True
        size = min(max(SEGMENT_BYTES, nbytes), self.capacity - self._allocated)
        if nbytes > size:
            return False
        self.segments.append(torch.empty(size, dtype=torch.uint8).share_memory_())
        self._allocated += size
        self._cursor = 0
        return True

    def filling(self) -> AbstractContextManager[None]:
        return self._filling_window()

    @contextmanager
    def _filling_window(self) -> Iterator[None]:
        if not self._may_fill:
            raise RuntimeError(
                "This cache reached another process, where its arena is already filled and shared. Fill it "
                "in the parent, before the workers start; a second filler would write over the first's bytes."
            )
        self._filling = True
        try:
            yield
        finally:
            self._filling = False

    @property
    def usage(self) -> CacheUsage:
        return CacheUsage(len(self._index), self._used, self._allocated, self.capacity, self._declined, self._full)

    def __getstate__(self) -> dict[str, object]:
        """A copy that crosses a process boundary is a reader: the arena it points at is already filled.

        Two processes filling one arena would each advance their own cursor over the other's bytes —
        decoded pixels quietly wrong rather than a crash. Reachable: under ``ddp_spawn`` Lightning runs
        ``setup()`` in every rank, so the copy refuses rather than trusting that nobody asks.
        """
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_filling"] = False
        state["_may_fill"] = False
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()


def _nbytes(entry: _Entry) -> int:
    return int(np.prod(entry.shape)) * np.dtype(entry.dtype).itemsize
