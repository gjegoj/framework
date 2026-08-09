"""Stitching: four samples become one picture, split once across the batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core.entities import Batch, as_tensor
from src.core.taxonomy import Modality, Objective, Topology
from src.transforms.batch.labels import as_soft, class_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.entities import DataProfile, Task


class Mosaic:
    """A 2x2 stitch across the batch, composing every task's target exactly.

    Quadrant *k* takes its pixels from the batch rolled by *k*, top-left keeping the
    sample's own — no resize, so every pixel comes from exactly one source. That is
    what separates stitching from blending, and what lets a segmentation mask compose
    by the identical swap and stay a valid index map: no interpolation, no dtype to
    juggle. A global label instead takes the four quadrant areas as its weights, the
    way a pasted rectangle's area weights a CutMix label.

    A batch shorter than four wraps around, which costs variety but stays correct: the
    label weights follow the same rolls, so a sample still reports where its pixels came
    from. One split point per batch keeps the whole thing to a clone and three slice
    assignments.

    Parameters:
        tasks (Sequence[Task]): Every task whose target must be rewritten.
        profile (DataProfile): Where the class counts come from.
        input_name (str): Which input holds the image.
        split_range (tuple[float, float]): Where the split may fall, as a
            fraction of height and width. Sampled once per batch, separately
            for each axis. Widening it towards 0 and 1 makes lopsided quadrants
            more likely.
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        profile: DataProfile,
        input_name: str = Modality.IMAGE,
        split_range: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        low, high = split_range
        if not 0.0 < low <= high < 1.0:
            raise ValueError(f"Mosaic needs 0 < low <= high < 1 for split_range, got {split_range}.")
        refused = [
            task.name
            for task in tasks
            if task.topology not in {Topology.GLOBAL, Topology.DENSE} or task.objective is Objective.METRIC
        ]
        if refused:
            raise ValueError(
                f"Mosaic cannot rewrite the targets of {', '.join(refused)}: it composes a picture and "
                f"whatever is laid over it, so a task without one has nothing to compose, and soft "
                f"labels break metric learning. Drop the transform, or the task it cannot serve."
            )
        # A mask is swapped like the picture; a label is weighted by the four areas.
        self._masks = [task.name for task in tasks if task.topology is Topology.DENSE]
        self._classes = class_counts([task for task in tasks if task.topology is Topology.GLOBAL], profile)
        self._input_name = input_name
        self._split_range = split_range

    def __call__(self, batch: Batch) -> Batch:
        """Return a new batch; the one given is never written into."""
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
                        as_tensor(batch.targets[name], task=name, wanted_by="a batch transform"), split_y, split_x
                    )
                    for name in self._masks
                },
                **{
                    name: self._weigh(
                        as_tensor(batch.targets[name], task=name, wanted_by="a batch transform"), name, shares
                    )
                    for name in self._classes
                },
            },
            meta=batch.meta,
        )

    def _split(self, size: int) -> int:
        low = max(1, int(self._split_range[0] * size))
        high = max(low + 1, int(self._split_range[1] * size))
        return int(torch.randint(low, high, (1,)).item())

    def _weigh(self, label: Tensor, name: str, shares: Sequence[float]) -> Tensor:
        soft = as_soft(label, self._classes[name])
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
