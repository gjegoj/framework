"""Stateless transformations remain functions; target geometry belongs to their adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from src.core import Geometry, Sample

type SampleTransform = Callable[[Sample], Sample]


@runtime_checkable
class GeometryAware(Protocol):
    """A transform that moves pixels asks which values move with them; the data build answers from the encoders."""

    def with_geometry(
        self,
        inputs: Mapping[str, Geometry],
        targets: Mapping[str, Geometry],
        auxiliary_inputs: Mapping[str, Geometry],
    ) -> SampleTransform: ...
