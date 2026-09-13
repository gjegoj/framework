"""albumentations behind the sample seam: one pipeline call moves the image and its masks together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import albumentations as A

from src.core import Geometry, Role, Sample
from src.transforms.base import AnswersTask, SampleTransform

PIPELINE_KIND = {Geometry.IMAGE: "image", Geometry.MASK: "mask"}
"""How a geometry travels through ``albumentations.Compose``; a value whose geometry is absent here stays put.

Which is the whole of the rule, and the reason there is no refusal beside it: ``Geometry`` names what
a raw value is, and the two entries here are everything it can be besides ``NONE``. A third kind —
boxes, keypoints — arrives with the ``Compose`` argument it travels as, and adds its line here."""

LABEL = "label"
"""How a value that is not pixels travels, for the one case there is: a target an augmentation writes."""

type Roles = Mapping[str, Mapping[str, Any]]


class AlbumentationsTransform:
    """A declared pipeline; ``with_geometry`` binds it to what the encoders say moves with the image.

    ``additional_targets`` are derived from those geometries, never declared: the keys a pipeline
    registers cannot then contradict the values it is handed. Measured on albumentationsx 2.3.7:
    ``Compose`` needs no argument named ``image``, so the image keeps the name its input has.
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
        answered = _answered_tasks(self.transforms, targets)
        carried: Roles = {
            role: {
                name: kind
                for name, geometry in declared.items()
                if (kind := _travels_as(name, geometry, answered)) is not None
            }
            for role, declared in roles.items()
        }
        registered = {name: kind for held in carried.values() for name, kind in held.items()}
        if "image" not in registered.values():
            raise ValueError("An albumentations pipeline needs at least one image input to move.")
        options: dict[str, Any] = {"telemetry": False, **self.compose_options}
        return BoundPipeline(A.Compose(self.transforms, additional_targets=registered, **options), carried)


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


def _answered_tasks(transforms: Sequence[Any], targets: Mapping[str, Geometry]) -> frozenset[str]:
    """The tasks the pipeline's own augmentations answer — derived from them, never declared beside them.

    Measured on albumentationsx 2.3.7: a pipeline routes a value by its kind, and every value of one
    kind is put through every rule for it. Two writers in one pipeline would therefore each rewrite
    the other's target, so the second one is refused rather than named as a caveat.
    """
    answered = sorted(one.task for one in transforms if isinstance(one, AnswersTask))
    if len(answered) > 1:
        raise ValueError(
            f"{answered} are answered by augmentations of one pipeline, which routes a value by its kind "
            f"and so cannot tell two targets apart: each would rewrite the other's. One stage declares "
            f"one such augmentation."
        )
    for name in answered:
        if name not in targets:
            declared = ", ".join(sorted(targets)) or "none"
            raise ValueError(
                f"An augmentation of this pipeline answers the task {name!r}, which this run does not "
                f"declare, so it would write nothing for the length of the run. Declared tasks: {declared}."
            )
        if targets[name] is not Geometry.NONE:
            raise ValueError(
                f"An augmentation of this pipeline answers {name!r}, whose target is pixels "
                f"({targets[name].value}): pixels move with the image, and an answer is written over."
            )
    return frozenset(answered)


def _travels_as(name: str, geometry: Geometry, answered: frozenset[str]) -> str | None:
    """The kind a value reaches the pipeline as, or ``None`` for one that stays where it is.

    Everything a pipeline moves is pixels, so a class or a number rides along untouched — with the one
    exception this returns ``LABEL`` for: the target of a task an augmentation of this very pipeline
    answers has to reach it in order to be written.
    """
    return LABEL if name in answered else PIPELINE_KIND.get(geometry)


def _refuse_a_name_under_two_roles(roles: Roles) -> None:
    declared = [name for role in roles.values() for name in role]
    duplicated = sorted({name for name in declared if declared.count(name) > 1})
    if duplicated:
        raise ValueError(f"{duplicated} declared under more than one role; each name is one pipeline argument.")
