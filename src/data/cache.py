"""In-RAM cache of decoded images/masks to skip disk I/O + decode during training.

Memory-safety model (why this avoids the classic DataLoader blow-up):
- the cache is warmed ONCE in the parent process (``DataModule.setup``), before
  ``DataLoader`` forks its workers;
- it is READ-ONLY afterwards — workers never write to a cached array, so the
  pixel buffers stay shared across forks via copy-on-write (no per-worker copy);
- a byte budget caps total RAM so warm-up can't OOM.

``CachingLoader`` / ``CachingTargetEncoder`` are thin decorators (the ``Dataset`` stays
oblivious to caching). The store is filled only by ``ArrayCache.warm``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from torch import Tensor

from src.data.encoders import TargetEncoder
from src.data.loaders import InputLoader

log = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024**3


class ArrayCache:
    """Thread-safe, byte-capped ``path -> ndarray`` store, warmed in the parent.

    Parameters:
        max_bytes (int): Total RAM budget; ``<= 0`` disables the cache entirely.
    """

    def __init__(self, max_bytes: int) -> None:
        self._store: dict[str, np.ndarray] = {}
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = Lock()

    def warm(self, keys: Iterable[str], load: Callable[[str], np.ndarray], workers: int) -> None:
        """Pre-load ``keys`` via ``load`` (threaded) until the byte budget is hit.

        Already-cached keys are skipped, so this is safe to call once per loader
        against a single shared cache. A failing ``load`` (bad file) is logged
        and skipped, never fatal.
        """
        if self._max_bytes <= 0:
            return
        pending = [key for key in dict.fromkeys(keys) if key not in self._store]
        if not pending:
            return
        columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"),
        )
        with ThreadPoolExecutor(max_workers=workers) as pool, Progress(*columns) as progress:
            task = progress.add_task("Warming cache", total=len(pending), status=self._status())
            futures = {pool.submit(self._safe_load, load, key): key for key in pending}
            for future in as_completed(futures):
                key, array = future.result()
                if array is not None:
                    with self._lock:
                        if self._bytes + int(array.nbytes) <= self._max_bytes:
                            self._store[key] = array
                            self._bytes += int(array.nbytes)
                progress.update(task, advance=1, status=self._status())

    def _status(self) -> str:
        """Live progress suffix: total cached files and memory (cumulative across warms)."""
        return f"[cyan]{len(self._store)} files · {self._bytes / _BYTES_PER_GIB:.2f} GiB"

    @staticmethod
    def _safe_load(load: Callable[[str], np.ndarray], key: str) -> tuple[str, np.ndarray | None]:
        try:
            return key, load(key)
        except Exception as error:  # noqa: BLE001 — a bad file must not abort warm-up
            log.warning("Cache warm-up skipped %s: %s", key, error)
            return key, None

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached array for ``key`` or ``None`` (read-only, fork-safe)."""
        return self._store.get(key)

    @property
    def nbytes(self) -> int:
        """Total bytes currently cached."""
        return self._bytes

    def __len__(self) -> int:
        """Number of items currently cached."""
        return len(self._store)


@dataclass
class CachingLoader(InputLoader):
    """Decorator: serve an input loader's result from an ``ArrayCache`` on hit.

    Parameters:
        inner (InputLoader): The wrapped loader (e.g. ``ImageLoader``).
        cache (ArrayCache): Shared cache, keyed by the resolved path.
    """

    inner: InputLoader
    cache: ArrayCache

    def __post_init__(self) -> None:
        self.file_based = self.inner.file_based

    def load(self, value: str) -> Any:
        hit = self.cache.get(value)
        return hit if hit is not None else self.inner.load(value)


@dataclass
class CachingTargetEncoder(TargetEncoder):
    """Decorator: serve a spatial target encoder's ``load`` from an ``ArrayCache``.

    Delegates ``fit`` / ``to_tensor`` / ``num_classes`` / ``spatial`` to the inner
    encoder; only ``load`` (the file read) is cached.

    Parameters:
        inner (TargetEncoder): The wrapped encoder (e.g. ``MaskEncoder``).
        cache (ArrayCache): Shared cache, keyed by the resolved path.
    """

    inner: TargetEncoder
    cache: ArrayCache

    def __post_init__(self) -> None:
        self.spatial = self.inner.spatial

    def fit(self, values: Iterable[Any]) -> None:
        self.inner.fit(values)

    def load(self, value: Any) -> Any:
        hit = self.cache.get(value)
        return hit if hit is not None else self.inner.load(value)

    def to_tensor(self, value: Any) -> Tensor:
        return self.inner.to_tensor(value)

    @property
    def num_classes(self) -> int | None:
        return self.inner.num_classes
