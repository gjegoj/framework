"""``Task``: a named instance of a kind, with its facts, weight and rate."""

from __future__ import annotations

import dataclasses

import pytest

from src.core import TaskFacts
from src.tasks import Classification, Task
from tests.support.entities import a_task


def test_a_task_is_a_kind_with_a_name_and_the_facts_the_data_revealed() -> None:
    task = Task(name="species", kind=Classification(), facts=TaskFacts(num_classes=2, class_names=("cat", "dog")))

    assert task.facts.class_names == ("cat", "dog")
    assert task.weight == 1.0 and task.lr is None


def test_task_defaults_weight_to_one() -> None:
    assert a_task().weight == 1.0


def test_task_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        a_task(weight=0.0)


def test_task_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        a_task(name="  ")


def test_task_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="lr"):
        a_task(lr=0.0)


def test_task_is_immutable() -> None:
    task = a_task()

    with pytest.raises(dataclasses.FrozenInstanceError):
        task.weight = 2.0  # type: ignore[misc]
