"""``TaskConfig``: presets resolve to axes; contradictions and typos fail loudly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import TaskConfig
from src.core import Objective, OutputTopology


def test_preset_resolves_to_explicit_axes() -> None:
    task = TaskConfig.model_validate({"preset": "classification", "target": "label"})

    assert task.output_topology is OutputTopology.GLOBAL
    assert task.objective is Objective.MULTICLASS
    assert task.target == "label"


def test_explicit_axes_work_without_a_preset() -> None:
    task = TaskConfig.model_validate({"output_topology": "global", "objective": "continuous", "target": "age"})

    assert task.output_topology is OutputTopology.GLOBAL
    assert task.objective is Objective.CONTINUOUS


def test_preset_and_explicit_axes_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="not both"):
        TaskConfig.model_validate({"preset": "classification", "objective": "binary"})


def test_axes_are_required_when_no_preset_is_given() -> None:
    with pytest.raises(ValidationError, match="output_topology"):
        TaskConfig.model_validate({"target": "label"})


def test_unknown_preset_lists_the_known_ones() -> None:
    with pytest.raises(ValidationError, match="classification"):
        TaskConfig.model_validate({"preset": "object_finding"})


def test_non_positive_weight_is_rejected() -> None:
    with pytest.raises(ValidationError, match="weight"):
        TaskConfig.model_validate({"preset": "classification", "weight": 0})


def test_a_typo_in_a_task_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="looss"):
        TaskConfig.model_validate({"preset": "classification", "looss": "ce"})


def test_declared_classes_must_cover_a_contiguous_index_range() -> None:
    """A gap means a head output nobody named; catching it at load beats a silent mislabel."""
    with pytest.raises(ValidationError, match="missing: 1"):
        TaskConfig.model_validate({"preset": "classification", "target": "t", "classes": {0: "cat", 2: "dog"}})


def test_declared_classes_refuse_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="duplicated"):
        TaskConfig.model_validate({"preset": "classification", "target": "t", "classes": {0: "cat", 1: "cat"}})


def test_classes_on_a_continuous_objective_are_refused() -> None:
    """Bins own a continuous target's value space; a vocabulary there is a contradiction."""
    with pytest.raises(ValidationError, match="continuous"):
        TaskConfig.model_validate({"preset": "regression", "target": "t", "classes": {0: "low"}})


def test_declared_classes_arrive_typed() -> None:
    task = TaskConfig.model_validate({"preset": "classification", "target": "t", "classes": {0: "cat", 1: "dog"}})

    assert task.classes == {0: "cat", 1: "dog"}
