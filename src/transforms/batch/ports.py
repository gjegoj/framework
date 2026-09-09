"""The contract a batch transform satisfies, for the callback that applies one."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Container

    from src.core.entities import Batch
    from src.core.taxonomy import OutputTopology
    from src.tasks import Task


@runtime_checkable
class BatchTransform(Protocol):
    """Transforms one collated batch — the seam for augmentations that mix samples.

    A ``SampleTransform`` cannot do this: while one sample is being loaded, the samples it
    would mix with do not exist yet. Mixing rewrites every task's target, so the transform
    needs the run's tasks — and the module is where they are declared. It is built from
    config without them and bound by the callback when the trainer sets the run up, so a
    task it cannot serve is refused before the first batch, by name.

    Lives here rather than in ``core/ports.py`` because it names ``Task``, which the core
    does not know: a port sits with the lowest package that can spell its contract.
    """

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        """The callable that does the work, bound to these tasks.

        Returns a new ``Batch`` rather than mutating, because the callback that applies
        one is the single place a batch is written into. A transform that needs no tasks
        returns itself.
        """
        ...


def refuse_unservable(
    transform: object, tasks: Sequence[Task], shapes: Container[OutputTopology], because: str
) -> None:
    """Refuse, by name, every task whose target ``transform`` cannot rewrite.

    One refusal for the mixing transforms, so it reads the same whichever raised it: a task
    is unservable when its output shape is not one the transform composes, or when its kind
    says soft targets break it (metric learning). ``because`` is the transform's own reason.
    """
    refused = [task.name for task in tasks if task.kind.shape not in shapes or not task.kind.mixable]
    if refused:
        raise ValueError(
            f"{type(transform).__name__} cannot rewrite the targets of {', '.join(refused)}: {because} "
            f"Drop the transform, or the task it cannot serve."
        )


def unbound(transform: object) -> RuntimeError:
    """The error a transform raises when handed a batch before ``for_tasks`` bound it."""
    return RuntimeError(
        f"{type(transform).__name__} was given a batch before being bound to the run's tasks: "
        f"call for_tasks(tasks) first, as the batch_transform callback does at setup."
    )
