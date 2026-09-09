"""Core entities a test hands a builder, instead of building the layer that produces them.

``kind.components(task, backbone)`` needs a ``Task`` carrying its facts. A test about
*building* should state those facts and nothing else — writing a dataset and running
``setup`` to obtain them would make the assertion depend on a fixture rather than on the
numbers the test named.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core import TaskFacts
from src.models.composite import AdaptedTarget
from src.tasks import Classification, Task, TaskKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def as_is(target: Tensor) -> AdaptedTarget:
    """Both views are the raw target — the adapter a test uses when adapting is not its subject.

    It lived in ``src/tasks/adapters.py`` beside the four the objectives actually choose,
    and had no production caller at all: every objective picks ``as_class_indices``,
    ``as_indicators``, ``float_for_loss`` or ``expectation_of``. A fixture in the shipped
    package reads as a fifth supported adapter, so it moved to where its callers are.
    """
    return AdaptedTarget(for_loss=target, for_metrics=target)


CLASS_NAMES = ["dog", "cat"]
"""The vocabulary ``write_dataset`` produces, in the order a fitted encoder learns it."""


def a_task(
    name: str = "label",
    kind: TaskKind | None = None,
    *,
    facts: TaskFacts | None = None,
    class_names: Sequence[str] | None = None,
    **overrides: Any,
) -> Task:
    """One classification task named ``label``, with whatever the test changes about it.

    The defaults are the least a ``Task`` needs to exist, so an override is always the
    thing under test: a kind, its facts, a weight, a rate. ``class_names`` is the short
    spelling of facts that carry a vocabulary.
    """
    if facts is None:
        facts = TaskFacts(num_classes=len(class_names), class_names=tuple(class_names)) if class_names else TaskFacts()
    return Task(name=name, kind=kind if kind is not None else Classification(), facts=facts, **overrides)


def dataset_facts(**tasks: int | TaskFacts) -> dict[str, TaskFacts]:
    """What ``setup()`` returns, for a test that skips the pipeline: a class count per task, or the facts whole.

    ``dataset_facts()`` is the one-task default every wiring test builds on; ``dataset_facts(label=3,
    mask=2)`` several; ``dataset_facts(label=TaskFacts(num_classes=2, class_names=[...]))`` where the
    names matter.
    """
    declared = tasks or {"label": 2}
    return {name: TaskFacts(num_classes=value) if isinstance(value, int) else value for name, value in declared.items()}
