"""Stitching: four samples become one picture, split once across the batch."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

from src.core.entities import Batch, require_tensor
from src.core.taxonomy import Modality, OutputTopology
from src.transforms.batch.ports import refuse_unservable, unbound

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from src.tasks import Task


class Mosaic:
    """A 2x2 stitch across the batch, composing every task's target exactly.

    Quadrant *k* takes its pixels from the batch rolled by *k* — no resize, so every pixel
    comes from exactly one source and a segmentation mask composes by the same swap. A
    global label takes the four quadrant areas as its weights. A batch shorter than four
    wraps around, which costs variety but stays correct. A task it cannot serve is refused
    when the tasks are bound (``for_tasks``), before the first batch.

    Parameters:
        input_name (str): Which input holds the image.
        split_range (tuple[float, float]): Where the split may fall, as a fraction of height
            and width; sampled once per batch, separately per axis.
    """

    def __init__(self, input_name: str = Modality.IMAGE, split_range: tuple[float, float] = (0.3, 0.7)) -> None:
        low, high = split_range
        if not 0.0 < low <= high < 1.0:
            raise ValueError(f"Mosaic needs 0 < low <= high < 1 for split_range, got {split_range}.")
        self._masks: list[str] | None = None
        self._labels: dict[str, Task] = {}
        self._input_name = input_name
        self._split_range = split_range

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        """A copy bound to these tasks — the ``BatchTransform`` port.

        A copy rather than a rebinding in place, so the declared transform stays what
        config said and the unbound refusal below stays honest for it. Each task knows how
        its own target softens and carries the facts that takes.
        """
        refuse_unservable(
            self,
            tasks,
            {OutputTopology.GLOBAL, OutputTopology.DENSE},
            "it composes a picture and whatever is laid over it, so a task without one has nothing to "
            "compose, and soft labels break metric learning.",
        )
        bound = copy.copy(self)
        # A mask is swapped like the picture; a label is weighted by the four areas.
        bound._masks = [task.name for task in tasks if task.kind.shape is OutputTopology.DENSE]
        bound._labels = {task.name: task for task in tasks if task.kind.shape is OutputTopology.GLOBAL}
        return bound

    def __call__(self, batch: Batch) -> Batch:
        """Return a new batch; the one given is never written into."""
        if self._masks is None:
            raise unbound(self)
        image = batch.inputs[self._input_name]
        height, width = image.shape[-2:]
        split_y, split_x = self._split(height), self._split(width)
        # In quadrant order, which is roll order: top left, top right, bottom left, bottom right.
        shares = [y * x / (height * width) for y in (split_y, height - split_y) for x in (split_x, width - split_x)]
        return Batch(
            inputs={**batch.inputs, self._input_name: _stitch(image, split_y, split_x)},
            targets={
                **batch.targets,
                **{
                    name: _stitch(
                        require_tensor(batch.targets[name], task=name, wanted_by="a batch transform"), split_y, split_x
                    )
                    for name in self._masks
                },
                **{
                    name: self._weigh(
                        require_tensor(batch.targets[name], task=name, wanted_by="a batch transform"), task, shares
                    )
                    for name, task in self._labels.items()
                },
            },
            meta=batch.meta,
        )

    def _split(self, size: int) -> int:
        low = max(1, int(self._split_range[0] * size))
        high = max(low + 1, int(self._split_range[1] * size))
        return int(torch.randint(low, high, (1,)).item())

    def _weigh(self, label: Tensor, task: Task, shares: Sequence[float]) -> Tensor:
        soft = task.kind.soften(label, task.facts)
        return sum((share * soft.roll(k, 0) for k, share in enumerate(shares)), start=torch.zeros_like(soft))


def _stitch(composed: Tensor, split_y: int, split_x: int) -> Tensor:
    """Swap three quadrants in from rolled neighbours; height and width come last.

    Slicing before rolling is what keeps this cheap: the roll then copies one
    quadrant window across the batch rather than the whole tensor. The trailing
    dimensions serve images ``[B, C, H, W]`` and masks ``[B, H, W]`` alike.
    """
    stitched = composed.clone()
    stitched[..., :split_y, split_x:] = composed[..., :split_y, split_x:].roll(1, 0)  # top right
    stitched[..., split_y:, :split_x] = composed[..., split_y:, :split_x].roll(2, 0)  # bottom left
    stitched[..., split_y:, split_x:] = composed[..., split_y:, split_x:].roll(3, 0)  # bottom right
    return stitched
