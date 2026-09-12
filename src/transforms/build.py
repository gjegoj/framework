"""The transforms section, turned into one sample transform per stage."""

from __future__ import annotations

from collections.abc import Mapping

from src.config import ComponentConfig
from src.config.instantiate import instantiate
from src.core import Geometry, Stage
from src.transforms.base import GeometryAware, SampleTransform


def build_transforms(
    declared: Mapping[Stage, ComponentConfig], geometries: Mapping[str, Mapping[str, Geometry]]
) -> dict[str, SampleTransform]:
    """One sample transform per stage, keyed by the split that runs it — the convention ``Stage`` declares.

    Keyed by a plain name rather than by ``Stage``, because that is how it is read: a data module serves
    arbitrary split names and looks one up by the name of the split it is preparing. ``Stage`` being a
    ``StrEnum`` is what lets the two meet — a stage *is* its name — and is why a split may be called
    anything while the stages a run declares transforms for stay a closed set.

    A transform that moves pixels is bound here to what the encoders publish about each value; one that
    does not is taken as declared. There is no registry to reach past: a chain is written out in full,
    under ``_target_``, because what a stage does to an image is the stage's own statement.
    """
    built: dict[str, SampleTransform] = {}
    for stage, component in declared.items():
        transform = instantiate(component)
        built[stage] = transform.with_geometry(**geometries) if isinstance(transform, GeometryAware) else transform
    return built
