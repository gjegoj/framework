"""The transforms section becomes one callable per stage, and a pixel one is told what moves with it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from inspect import signature
from typing import Any

from src.config import ComponentConfig
from src.core import Geometry, Role, Sample, Stage
from src.transforms.base import GeometryAware
from src.transforms.build import build_transforms

GEOMETRIES = {"inputs": {"image": Geometry.IMAGE}, "targets": {"mask": Geometry.MASK}, "auxiliary_inputs": {}}


class Flip:
    """A transform that records what it was bound to, so the binding itself can be read back."""

    bound: Mapping[str, Mapping[str, Geometry]] = {}

    def with_geometry(
        self, inputs: Mapping[str, Geometry], targets: Mapping[str, Geometry], auxiliary_inputs: Mapping[str, Geometry]
    ) -> Any:
        type(self).bound = {
            "inputs": dict(inputs),
            "targets": dict(targets),
            "auxiliary_inputs": dict(auxiliary_inputs),
        }
        return lambda sample: sample

    def __call__(self, sample: Sample) -> Sample:
        return sample


def test_transforms_are_keyed_by_split_and_told_the_geometry() -> None:
    declared = {Stage.TRAIN: ComponentConfig.model_validate({"_target_": "tests.unit.transforms.test_build.Flip"})}

    built = build_transforms(declared, GEOMETRIES)

    assert set(built) == {"train"} and Flip.bound == GEOMETRIES


def test_the_three_places_a_sample_carries_values_are_named_once() -> None:
    """``Role`` says its values *are* ``Sample``'s field names, which is what lets a preprocessor's
    ``geometries`` reach ``with_geometry`` as keyword arguments without either side spelling them again.

    Nothing in the type system holds that: the mapping passed through is keyed by ``str``, so renaming a
    role — or a field, or a parameter — lands as a bare ``TypeError`` about an unexpected keyword the
    first time a run builds its transforms, with all three spellings looking correct where they are read.
    """
    named = {one.value for one in Role}

    assert named <= {one.name for one in fields(Sample)}, "a role that is not a field the sample carries"
    assert named == set(signature(GeometryAware.with_geometry).parameters) - {"self"}
