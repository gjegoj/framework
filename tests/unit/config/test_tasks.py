"""``TaskConfig``: a task is declared by its kind; contradictions and typos fail loudly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import TaskConfig


def test_a_kind_is_declared_by_its_familiar_name() -> None:
    task = TaskConfig.model_validate({"kind": "classification", "target": "label"})

    assert task.kind.name == "classification"
    assert task.target == "label"


def test_a_kind_of_your_own_is_declared_by_import_path() -> None:
    task = TaskConfig.model_validate({"kind": {"_target_": "my_pkg.Depth"}, "target": "depth"})

    assert task.kind.target == "my_pkg.Depth"
    assert task.kind.name is None


def test_the_kind_is_required() -> None:
    with pytest.raises(ValidationError, match="kind"):
        TaskConfig.model_validate({"target": "label"})


@pytest.mark.parametrize("retired", ["preset", "output_topology", "input_topology", "objective"])
def test_a_retired_spelling_is_refused_naming_kind(retired: str) -> None:
    """The axes and the preset that stood for them are gone; a config written for them is told what to write."""
    with pytest.raises(ValidationError, match="'kind'"):
        TaskConfig.model_validate({retired: "classification", "target": "t"})


def test_non_positive_weight_is_rejected() -> None:
    with pytest.raises(ValidationError, match="weight"):
        TaskConfig.model_validate({"kind": "classification", "weight": 0})


def test_a_typo_in_a_task_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="looss"):
        TaskConfig.model_validate({"kind": "classification", "looss": "ce"})


def test_declared_classes_arrive_typed() -> None:
    task = TaskConfig.model_validate({"kind": "classification", "target": "t", "classes": {0: "cat", 1: "dog"}})

    assert task.classes == {0: "cat", 1: "dog"}


def test_metrics_left_undeclared_stay_undeclared_until_the_build() -> None:
    """The kind's defaults are a fact of ``tasks/``, which config does not read; the composition root fills them in."""
    task = TaskConfig.model_validate({"kind": "classification", "target": "t"})

    assert task.metrics is None


@pytest.mark.parametrize(("declared", "expected"), [("encoder", ("encoder",)), (["p4", "p5"], ("p4", "p5"))])
def test_streams_take_one_name_or_several_and_read_back_as_a_tuple(declared: object, expected: tuple[str, ...]) -> None:
    task = TaskConfig.model_validate(
        {"kind": "classification", "target": "t", "classes": {0: "a", 1: "b"}, "streams": declared}
    )

    assert task.streams == expected


def test_the_old_stream_key_is_refused_naming_the_rename() -> None:
    with pytest.raises(ValidationError, match="stream"):
        TaskConfig.model_validate(
            {"kind": "classification", "target": "t", "classes": {0: "a", 1: "b"}, "stream": "encoder"}
        )


def test_native_head_is_no_longer_a_key() -> None:
    """One key chooses the head: ``head: {name: native}`` keeps the backbone's own; a second key could only disagree."""
    with pytest.raises(ValidationError, match="native_head"):
        TaskConfig.model_validate({"kind": "classification", "target": "label", "native_head": True})
