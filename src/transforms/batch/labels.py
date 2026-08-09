"""What a batch transform has to do to a target before two of them can be combined."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn.functional import one_hot

from src.core.taxonomy import Objective

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.entities import DataProfile, Task


def class_counts(tasks: Sequence[Task], profile: DataProfile) -> dict[str, int | None]:
    """How wide each task's target becomes when softened; ``None`` where it already is a number.

    Only class *indices* have to be widened: a price, an indicator vector or a
    distribution over bins is already something a weighted sum can be taken of. Which of
    the two a task is follows from its objective, and the width from the profile — so a
    transform states neither.

    Shared because both transforms that combine targets built exactly this map, the same
    way, down to a near-identical comment. What they do *not* share is which tasks they
    can serve at all — a blend has no coherent per-pixel target while a stitch does — so
    each filters ``tasks`` itself and refuses the rest in its own words, where the reason
    belongs.
    """
    return {
        task.name: profile.require_num_classes(task.name) if task.objective is Objective.MULTICLASS else None
        for task in tasks
    }


def as_soft(label: Tensor, classes: int | None) -> Tensor:
    """One target in the shape a weighted sum of two targets needs.

    Class indices widen to one-hot at ``classes``; anything else is already a number and
    only wants a float dtype.
    """
    return one_hot(label.long(), classes).float() if classes is not None else label.float()


__all__ = ["as_soft", "class_counts"]
