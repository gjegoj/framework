"""Holding decoded files in RAM, so an epoch does not decode what the last one did."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING, Any, override

import numpy as np

from src.data.registry import cache_registry
from src.progress import track

if TYPE_CHECKING:
    from src.data.loaders import InputLoader

log = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3

NAMESPACE_SEPARATOR = "\0"
"""Divides a namespace from the cell value — the one byte a filename cannot hold."""


class LoaderCache(ABC):
    """Keeps what a loader returned, so the next epoch does not read it again.

    Keys are cell values as the annotation table stores them, never resolved paths —
    the loader keeps sole ownership of how a relative path becomes absolute.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """The cached value for ``key``, or ``None`` when it is not held."""

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Offer a value for caching.

        A hint, not an instruction: an implementation may decline anything — a
        value of the wrong kind, one that does not fit, or any value at all once
        it has stopped accepting writes.
        """

    @abstractmethod
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any]) -> None:
        """Fill the cache by calling ``load`` once per distinct key.

        ``warm`` stores nothing itself — it drives a loader wrapped with
        :func:`cached`, and that wrapper is the one line of code which writes to
        the store; it also serves a value it already holds without loading, so
        warming twice is cheap rather than wrong. Run this in the parent
        process, before any worker forks.
        """

    def scoped(self, namespace: str) -> LoaderCache:
        """A view of this cache whose keys are private to ``namespace``.

        One cache serves every input and every file-reading encoder, so that
        they share one budget — but a cell value alone does not say which column
        it came from. An image column and a mask column holding the same
        filename under different roots would otherwise serve each other's
        arrays, and the pixels would arrive where an index map was promised.

        A view object rather than a key convention at each call site: the one
        object carries the store and the identity together, so a consumer that
        is handed a cache cannot forget the namespace that makes keys its own.
        """
        return _ScopedCache(self, namespace)


class _ScopedCache(LoaderCache):
    """The keys of one namespace, held in the cache every namespace shares."""

    def __init__(self, inner: LoaderCache, namespace: str) -> None:
        self._inner = inner
        self._namespace = namespace

    @override
    def get(self, key: str) -> Any | None:
        return self._inner.get(self._scoped(key))

    @override
    def put(self, key: str, value: Any) -> None:
        self._inner.put(self._scoped(key), value)

    @override
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any]) -> None:
        # Keys pass through untranslated: ``load`` is a loader wrapped with
        # :func:`cached`, and that wrapper already serves a held value without
        # loading — warming never needs to know what a store key looks like.
        self._inner.warm(keys, load)

    def _scoped(self, key: str) -> str:
        return f"{self._namespace}{NAMESPACE_SEPARATOR}{key}"


@cache_registry.register("ram")
class RamCache(LoaderCache):
    """Decoded arrays held in memory, up to a byte budget.

    Only arrays are kept: a loader returning text or a scalar is simply not cached,
    which is why no loader has to declare whether it reads files.

    **Filled once in the parent process, read-only afterwards.** Writes are accepted
    only while ``warm`` runs, which is what keeps the store frozen once ``DataLoader``
    forks — a cache filled lazily inside workers would be filled independently by each
    of them, multiplying its size by ``num_workers``.

    Copy-on-write then carries the pixels but not the scaffolding. An array's data
    buffer is a separate allocation that reading does not reference-count, so the
    gigabytes are genuinely shared; the dict, its keys and the array object headers
    *are* reference-counted, and CPython writes to them on ordinary reads, so those
    pages are copied per worker — tens of megabytes at a hundred thousand entries.
    Removing that remainder needs shared memory or a memory-mapped file, which is a
    second implementation of this port rather than a change to this one.

    Parameters:
        max_gib (float): Memory budget. The cache stops taking values at it and
            the remaining files are read from disk each epoch, as they would be
            without a cache at all.
        workers (int): Threads used to warm it. Image decoding releases the GIL,
            so these genuinely overlap.
    """

    def __init__(self, max_gib: float = 4.0, workers: int = 8) -> None:
        if max_gib <= 0:
            raise ValueError(f"A ram cache needs a positive max_gib, got {max_gib}; omit the section to disable it.")
        if workers < 1:
            raise ValueError(f"A ram cache needs at least one worker, got {workers}.")
        self._max_bytes = int(max_gib * BYTES_PER_GIB)
        self._workers = workers
        self._store: dict[str, np.ndarray] = {}
        self._bytes = 0
        self._filling = False
        self._lock = Lock()

    @override
    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    @override
    def put(self, key: str, value: Any) -> None:
        if not self._filling or not isinstance(value, np.ndarray):
            return
        with self._lock:
            if key in self._store or self._bytes + int(value.nbytes) > self._max_bytes:
                return
            self._store[key] = value
            self._bytes += int(value.nbytes)

    @override
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any]) -> None:
        pending = [key for key in dict.fromkeys(keys) if key not in self._store]
        if not pending:
            return

        def read(key: str) -> None:
            try:
                load(key)
            except Exception as error:  # noqa: BLE001 — one unreadable file must not abort a run
                log.warning("Cache skipped '%s': %s", key, error)

        self._filling = True
        try:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                for _ in track(pool.map(read, pending), "Caching files", len(pending)):
                    pass
        finally:
            self._filling = False
        log.info(
            "Cache holds %d file(s), %.2f of %.2f GiB.",
            len(self._store),
            self._bytes / BYTES_PER_GIB,
            self._max_bytes / BYTES_PER_GIB,
        )


def cached(load: InputLoader, cache: LoaderCache) -> InputLoader:
    """Wrap a loader so its result is served from ``cache`` when it is held there.

    The only code in the framework that writes to a cache, so a value can never be
    stored under a key that means something else.
    """

    def read(value: Any) -> Any:
        key = str(value)
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = load(value)
        cache.put(key, result)
        return result

    return read
