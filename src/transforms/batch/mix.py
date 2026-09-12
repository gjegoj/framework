"""Blending and pasting: two samples become one picture, and their labels follow the same draw."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.distributions import Beta

from src.core import Batch, Modality, require_tensor

if TYPE_CHECKING:
    from src.tasks import Task

PAIRED_WITH = 1
"""How far a sample sits from the one it mixes with: the batch, rolled by one.

A shuffled loader is what makes that pairing arbitrary, which is what it has to be. Every label rolls
by the same amount, or a picture would take one neighbour's pixels and another neighbour's label.
"""


class LabelMix(ABC):
    """One draw mixes the picture, and by the same weight every task's label.

    Every task is rewritten from the *same* draw, because the one picture they all read changed once.
    Only tasks measured over the whole picture are served: a mixed picture has no coherent per-pixel
    target, and that is refused when the tasks are bound rather than at the first batch.

    What a subclass supplies is the mix itself — how the two pictures become one, and what share of
    the first survived it, which is the weight the labels then take.

    Parameters:
        alpha: The Beta parameter the weight is drawn from; the customary 1.0 draws every weight
            equally, and larger values crowd the draw towards an even mix.
        input_name: Which input holds the picture.
    """

    def __init__(self, alpha: float = 1.0, input_name: str = Modality.IMAGE) -> None:
        if alpha <= 0:
            raise ValueError(f"{type(self).__name__} draws its weight from Beta(alpha, alpha); alpha was {alpha}.")
        self._weights = Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))
        self._input_name = input_name

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        """The mix bound to these tasks — the ``BatchTransform`` contract.

        The binding lives in what is returned rather than in the object: a declaration builds one of
        these and keeps it, and an object that is a transform only *after* someone has bound it has a
        state in which calling it is a mistake. There is no such state here.
        """
        dense = [task.name for task in tasks if task.dense]
        if dense:
            raise ValueError(
                f"{type(self).__name__} makes one picture out of two, and such a picture has no coherent "
                f"per-pixel target: {', '.join(dense)} is measured at every pixel. Drop the transform, or "
                "the task it cannot serve."
            )
        return partial(self._applied, tasks={task.name: task for task in tasks})

    def _applied(self, batch: Batch, tasks: Mapping[str, Task]) -> Batch:
        """A new batch; the one handed over is never written into."""
        picture, weight = self._mixed(require_tensor(batch.inputs[self._input_name], name=self._input_name))
        return Batch(
            inputs={**batch.inputs, self._input_name: picture},
            targets={**batch.targets, **{name: _shared(task, batch, weight) for name, task in tasks.items()}},
            count=batch.count,
            metadata=batch.metadata,
        )

    def _drawn(self) -> float:
        return float(self._weights.sample())

    @abstractmethod
    def _mixed(self, picture: Tensor) -> tuple[Tensor, float]:
        """The picture mixed with its neighbour, and the share of itself it kept."""


class MixUp(LabelMix):
    """Blend two pictures, and their labels, by one drawn weight."""

    def _mixed(self, picture: Tensor) -> tuple[Tensor, float]:
        weight = self._drawn()
        return weight * picture + (1.0 - weight) * picture.roll(PAIRED_WITH, 0), weight


class CutMix(LabelMix):
    """Paste a patch of the neighbour over each picture, and weigh the labels by what is left of it.

    The weight the labels take is the area that stayed rather than the one drawn: a patch is clipped
    where it runs off the edge, so the two differ, and a label describing the drawn patch would
    describe a picture nobody was shown.
    """

    def _mixed(self, picture: Tensor) -> tuple[Tensor, float]:
        height, width = picture.shape[-2:]
        top, left, bottom, right = self._patch(height, width)
        mixed = picture.clone()
        mixed[..., top:bottom, left:right] = picture.roll(PAIRED_WITH, 0)[..., top:bottom, left:right]
        return mixed, 1.0 - (bottom - top) * (right - left) / (height * width)

    def _patch(self, height: int, width: int) -> tuple[int, int, int, int]:
        """A box of the drawn area, around a point anywhere in the picture and clipped at its edges."""
        half = 0.5 * math.sqrt(1.0 - self._drawn())
        rows, columns = int(half * height), int(half * width)
        centre_row, centre_column = int(torch.randint(height, ())), int(torch.randint(width, ()))
        return (
            max(centre_row - rows, 0),
            max(centre_column - columns, 0),
            min(centre_row + rows, height),
            min(centre_column + columns, width),
        )


def _shared(task: Task, batch: Batch, weight: float) -> Tensor:
    """One task's target, softened so it admits a weighted sum, then weighed against its neighbour's."""
    target = task.soften(task.target(batch))
    return weight * target + (1.0 - weight) * target.roll(PAIRED_WITH, 0)
