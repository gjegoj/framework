"""Core entities a test hands a builder, instead of assembling the layer that produces them.

``build_task_components(task, profile, backbone)`` needs a ``Task`` and a ``DataProfile``.
A test about *building* should state those two facts and nothing else — writing a dataset
and running ``setup`` to obtain them would make the assertion depend on a fixture rather
than on the numbers the test named.
"""

from __future__ import annotations

from typing import Any

from src.core import DataProfile, Objective, TargetFacts, Task, Topology

CLASS_NAMES = ["dog", "cat"]
"""The vocabulary ``write_dataset`` produces, in the order a fitted encoder learns it."""


def a_task(**overrides: Any) -> Task:
    """One global multiclass task named ``label``, with whatever the test changes about it.

    The defaults are the least a ``Task`` needs to exist, so an override is always the
    thing under test: a topology, an objective, a weight, a rate.
    """
    return Task(
        **{
            "name": "label",
            "topology": Topology.GLOBAL,
            "objective": Objective.MULTICLASS,
            "metrics": {},
        }
        | overrides
    )


def profiled(task: str = "label", classes: int = 2, names: list[str] | None = None) -> DataProfile:
    """A profile holding one task's facts, as fitting its encoder would have left them.

    Parameters:
        task (str): The task the facts belong to.
        classes (int): How many classes the encoder learned.
        names (list[str] | None): Their names, where the test's subject needs them;
            ``None`` leaves the profile class-name-free, as an undeclared vocabulary is.
    """
    return profiling(**{task: classes}) if names is None else _named(task, classes, names)


def profiling(**classes: int) -> DataProfile:
    """A profile holding several tasks' class counts — ``profiling(label=3, mask=2)``."""
    profile = DataProfile()
    for task, count in classes.items():
        profile.record(task, TargetFacts(num_classes=count))
    return profile


def _named(task: str, classes: int, names: list[str]) -> DataProfile:
    profile = DataProfile()
    profile.record(task, TargetFacts(num_classes=classes, class_names=names))
    return profile
