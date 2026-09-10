"""Stateless transformations remain functions; target geometry belongs to their adapters."""

from __future__ import annotations

from collections.abc import Callable

from src.core import Batch, Sample

type SampleTransform = Callable[[Sample], Sample]
type BatchTransform = Callable[[Batch], Batch]
