"""albumentations behind the sample seam: one pipeline call moves the picture and its masks together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import albumentations as A

from src.core import Geometry, Role, Sample
from src.transforms.base import SampleTransform

PIPELINE_KIND = {Geometry.IMAGE: "image", Geometry.MASK: "mask"}
"""How a geometry travels through ``albumentations.Compose``; a geometry absent here is not pixels and cannot travel."""

type Roles = Mapping[str, Mapping[str, Geometry]]


class AlbumentationsTransform:
    """A declared pipeline; ``with_geometry`` binds it to what the encoders say moves with the picture.

    ``additional_targets`` are derived from those geometries, never declared: the keys a pipeline
    registers cannot then contradict the values it is handed. Measured on albumentationsx 2.3.7:
    ``Compose`` needs no argument named ``image``, so the picture keeps the name its input has.
    """

    def __init__(self, transforms: Sequence[Any], **compose_options: Any) -> None:
        if "additional_targets" in compose_options:
            raise ValueError("'additional_targets' is derived from the encoders' geometries; do not declare it.")
        self.transforms = list(transforms)
        self.compose_options = compose_options

    def with_geometry(
        self, inputs: Mapping[str, Geometry], targets: Mapping[str, Geometry], auxiliary_inputs: Mapping[str, Geometry]
    ) -> SampleTransform:
        roles: Roles = {
            Role.INPUTS: dict(inputs),
            Role.AUXILIARY: dict(auxiliary_inputs),
            Role.TARGETS: dict(targets),
        }
        _refuse_a_name_under_two_roles(roles)
        carried = {
            name: _pipeline_kind(role, name, geometry)
            for role, declared in roles.items()
            for name, geometry in declared.items()
        }
        if "image" not in carried.values():
            raise ValueError("An albumentations pipeline needs at least one image input to move.")
        options: dict[str, Any] = {"telemetry": False, **self.compose_options}
        return BoundPipeline(A.Compose(self.transforms, additional_targets=carried, **options), roles)


class BoundPipeline:
    """The pipeline with its names resolved: an object, so spawned data workers receive it intact.

    Only declared names that the sample carries are moved; a target absent at inference is simply absent.
    """

    def __init__(self, pipeline: A.Compose, roles: Roles) -> None:
        self.pipeline = pipeline
        self.roles = {role: dict(names) for role, names in roles.items()}

    def __call__(self, sample: Sample) -> Sample:
        present = [(role, held, [n for n in self.roles[role] if n in held]) for role, held in _values(sample)]
        carried: dict[str, Any] = {name: held[name] for _, held, names in present for name in names}
        moved = self.pipeline(**carried)
        return replace(sample, **{role: {**held, **{n: moved[n] for n in names}} for role, held, names in present})


def _values(sample: Sample) -> tuple[tuple[str, Mapping[str, object]], ...]:
    return (
        (Role.INPUTS, sample.inputs),
        (Role.AUXILIARY, sample.auxiliary_inputs),
        (Role.TARGETS, sample.targets),
    )


def _pipeline_kind(role: str, name: str, geometry: Geometry) -> str:
    """Everything a pipeline moves is pixels; anything else reaches it by mistake, whatever its role."""
    if geometry not in PIPELINE_KIND:
        travels = ", ".join(sorted(PIPELINE_KIND.values()))
        raise ValueError(
            f"{role} {name!r} declares geometry {geometry.value!r}, but a pipeline moves pixels: {travels}."
        )
    return PIPELINE_KIND[geometry]


def _refuse_a_name_under_two_roles(roles: Roles) -> None:
    declared = [name for role in roles.values() for name in role]
    duplicated = sorted({name for name in declared if declared.count(name) > 1})
    if duplicated:
        raise ValueError(f"{duplicated} declared under more than one role; each name is one pipeline argument.")
