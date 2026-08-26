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
    """How wide each task's target becomes when softened; ``None`` where it is already a number.

    Only class indices have to be widened; an indicator vector or a distribution over bins
    already admits a weighted sum. Which a task is follows from its objective, the width
    from the profile. Each mixing transform filters ``tasks`` itself and refuses the rest.
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
