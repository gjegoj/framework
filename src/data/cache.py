"""Holding decoded files in RAM, so an epoch does not decode what the last one did."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """How full a cache is, in the terms its budget was declared in.

    ``declined`` counts only what the *budget* turned away — decoded and refused,
    or skipped once the budget had already filled mid-column. A value refused for
    its kind — a scalar target, a string — is not in it: those are never cached
    by design, and counting them would report a full cache that is not.

    ``full`` is the budget's own verdict: a file did not fit, and warming stopped
    reading. Distinct from ``declined > 0`` in meaning, not just in type — it is
    the fact the summary line keys on, where ``declined`` is only its size.
    """

    files: int
    used_bytes: int
    capacity_bytes: int
    declined: int
    full: bool


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
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any], label: str = "files") -> None:
        """Fill the cache by calling ``load`` once per distinct key.

        ``warm`` stores nothing itself — it drives a loader wrapped with
        :func:`cached`, and that wrapper is the one line of code which writes to
        the store; it also serves a value it already holds without loading, so
        warming twice is cheap rather than wrong. Run this in the parent
        process, before any worker forks.

        ``label`` names *what* is being warmed (``train: input/image``) — the one
        thing only the caller knows. It titles the progress bar and keys the
        per-label accounting ``summarize`` reports, so what the bar said and what
        the summary says are one string.
        """

    @abstractmethod
    def usage(self) -> CacheUsage:
        """What the cache holds right now, against the budget it was given."""

    @abstractmethod
    def summarize(self) -> None:
        """Log the warm-up's closing lines: what is held, who took it, what did not fit.

        The trigger is the caller's — only it knows when the last ``warm`` of a pass
        has run — but the content is the cache's own accounting, which is why this is
        a method here and not a log line in the caller: the numbers, their per-label
        breakdown and their formatting never leave the module that produces them.
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
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any], label: str = "files") -> None:
        # Keys pass through untranslated: ``load`` is a loader wrapped with
        # :func:`cached`, and that wrapper already serves a held value without
        # loading — warming never needs to know what a store key looks like.
        self._inner.warm(keys, load, label)

    @override
    def usage(self) -> CacheUsage:
        # The whole store's usage, not the namespace's slice: the budget is shared,
        # and a view that reported only its own keys would understate how full it is.
        return self._inner.usage()

    @override
    def summarize(self) -> None:
        self._inner.summarize()

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
        max_gib (float): Memory budget. The first file that does not fit ends the
            warm-up's *reading*, not just its storing — decoding files only to
            discard them is the cost a cache exists to remove. What did not make
            it in is read from disk each epoch, as it would be without a cache at
            all. Deliberately forgone: a smaller file behind the one that filled
            the budget might still have fit.
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
        self._declined = 0
        self._full = False
        self._filling = False
        self._taken: dict[str, int] = {}
        self._lock = Lock()

    @override
    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    @override
    def put(self, key: str, value: Any) -> None:
        if not self._filling or not isinstance(value, np.ndarray):
            return
        with self._lock:
            if key in self._store:
                return
            if self._bytes + int(value.nbytes) > self._max_bytes:
                # Counted apart from the kind-based refusals above: only these mean
                # "the budget ran out", which is the one fact worth a closing log line.
                # The flag is the stop order: a smaller file behind this one might
                # still have fit, but hunting for it means decoding everything and
                # discarding most of it — the exact cost warming exists to remove.
                self._declined += 1
                self._full = True
                return
            self._store[key] = value
            self._bytes += int(value.nbytes)

    @override
    def warm(self, keys: Iterable[str], load: Callable[[Any], Any], label: str = "files") -> None:
        # A full cache reads nothing more: whole later columns are skipped here, and
        # deliberately *not* added to ``declined`` — without decoding, whether those
        # cells are files or scalars is unknowable, and the count must not guess.
        if self._full:
            return
        # Skips what this store already holds *under the keys it is given*. That is the
        # whole of what it can do, and it is worth saying which case it does not reach:
        # a loader wrapped with :func:`cached` over a `scoped` view writes namespaced
        # keys, so a raw cell value never matches one and every key is walked again.
        # Nothing is re-read — the wrapper serves the held value — so the second pass
        # costs a dict lookup per key rather than a decode.
        pending = [key for key in dict.fromkeys(keys) if key not in self._store]
        if not pending:
            return

        def read(key: str) -> None:
            # Filling ended mid-column, so this key *is* a file that will not fit —
            # counted, unlike the wholesale skip above. Which keys land here rather
            # than in a decoded-then-refused ``put`` depends on thread timing, but
            # the column's total — pending minus stored — does not.
            if self._full:
                with self._lock:
                    self._declined += 1
                return
            try:
                load(key)
            except Exception as error:  # noqa: BLE001 — one unreadable file must not abort a run
                log.warning("Cache skipped '%s': %s", key, error)

        self._filling = True
        before = self._bytes
        try:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                for _ in track(pool.map(read, pending), f"Caching {label}", len(pending), status=self._quota_line):
                    pass
        finally:
            self._filling = False
            # Own accounting, measured where the bytes live — a caller reconstructing
            # this by sampling ``usage`` around calls was the leak this line closes.
            self._taken[label] = self._taken.get(label, 0) + self._bytes - before

    @override
    def usage(self) -> CacheUsage:
        return CacheUsage(
            files=len(self._store),
            used_bytes=self._bytes,
            capacity_bytes=self._max_bytes,
            declined=self._declined,
            full=self._full,
        )

    @override
    def summarize(self) -> None:
        breakdown = ", ".join(
            f"{label} {spent / BYTES_PER_GIB:.2f} GiB" for label, spent in self._taken.items() if spent > 0
        )
        log.info(
            "Cache holds %d file(s) — %.2f of %.2f GiB%s.",
            len(self._store),
            self._bytes / BYTES_PER_GIB,
            self._max_bytes / BYTES_PER_GIB,
            f" ({breakdown})" if breakdown else "",
        )
        if self._full:
            log.info(
                "Cache budget full: %d file(s) were turned away while warming, and everything "
                "after them was skipped without reading; all of it comes from disk each epoch.",
                self._declined,
            )

    def _quota_line(self) -> str:
        """The bar's running answer to "how much of my quota is spent"."""
        return f"{self._bytes / BYTES_PER_GIB:.1f}/{self._max_bytes / BYTES_PER_GIB:.1f} GiB"


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
