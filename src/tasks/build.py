"""A `tasks` declaration becomes the objects a run is built around: what each task is, and what serves it.

Everything here answers a question about one task's declaration, which is why it is in this package
rather than at the composition root: a data scientist adding a kind reads `tasks/` and finds the rules
their class is held to. What a task is *joined with* — its loss, its metrics, its head's widths — is
decided where both sides are in view, and that is the root.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.config import ComponentConfig, HeadConfig, TaskConfig
from src.config.instantiate import instantiate, resolve_factory
from src.core import DatasetInfo, TargetInfo, naming
from src.tasks.base import Task
from src.tasks.registry import task_registry


def build_task_kinds(declared: Mapping[str, TaskConfig]) -> dict[str, type[Task]]:
    """What each task *is*, resolved before anything is read: the data pipeline asks the class first.

    Which encoder reads a task's column is a fact of its kind, declared on the class, so the kinds are
    resolved once here and the instances built later, when the data can say what their targets hold.
    """
    kinds = {}
    for name, task in declared.items():
        with naming(f"tasks.{name}.kind"):
            kind = resolve_factory(task.kind, task_registry)
        if not (isinstance(kind, type) and issubclass(kind, Task)):
            raise TypeError(
                f"Task {name!r} declares kind {task.kind.spelled!r}, which is not a task: a kind says what "
                "its target means, which head serves it and what it is judged by."
            )
        kinds[name] = kind
    return kinds


def default_target_encoder(kind: type[Task]) -> ComponentConfig | None:
    """The encoder a kind reads its column with, where it names one; a run may declare another.

    Read through the same validation a run's own line goes through, so a kind writes its default in the
    grammar every component is written in — a registry name, or an import path for one this framework
    does not hold. Its head is already declared that way, and the asymmetry left a reader with a kind
    of their own either registering an encoder into this tree or repeating `target_encoder:` in every
    experiment, for a fact the kind itself knows.
    """
    declared = kind.default_target_encoder
    return None if declared is None else ComponentConfig.model_validate(declared)


def build_tasks(declared: Mapping[str, TaskConfig], info: DatasetInfo) -> dict[str, Task]:
    """The declared tasks as objects, each carrying what the data settled about its target."""
    built = {}
    for name, task in declared.items():
        with naming(f"tasks.{name}.kind"):
            built[name] = instantiate(
                task.kind,
                task_registry,
                name=name,
                info=info.targets.get(name, TargetInfo()),
                weight=task.weight,
                lr=task.lr,
            )
    return built


def head_for(declared: TaskConfig, task: Task) -> HeadConfig:
    """The head a run declared, reading the stream its kind reads where the declaration named none.

    The two halves of a head are decided by different parties: which stream a task reads follows from
    its topology — a dense task reads the decoder, a whole-sample one the pooled vector — while which
    kind of head reads it is the run's to choose. So `head: native` keeps the kind's stream, and
    `head: {name: native, stream: encoder}` overrides both.
    """
    default = HeadConfig.model_validate(dict(task.default_head))
    if declared.head is None:
        return default
    if declared.head.stream is not None:
        return declared.head
    return declared.head.model_copy(update={"stream": default.stream})
