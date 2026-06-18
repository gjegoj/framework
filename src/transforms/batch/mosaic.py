"""Mosaic batch transform — 2x2 regional swap across the batch.

Each output sample keeps its own top-left quadrant and takes the other three
quadrants from rolled batch neighbours at the **same** spatial positions (no
resize). Every pixel therefore comes from exactly one source, so a DENSE mask
target composes by the identical swap and stays a valid index map — no
interpolation, no dtype juggling. A single split point per batch keeps the op
fully vectorized (one ``clone`` + three slice assignments).
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.core.entities import Batch
from src.core.keys import IMAGE
from src.core.ports import BatchTransform
from src.tasks.taxonomy import Topology
from src.transforms.batch.registry import batch_transforms
from src.transforms.batch.spec import TargetSpec


@batch_transforms.register("mosaic")
class Mosaic(BatchTransform):
    """2x2 mosaic over the batch, composing every DENSE mask target.

    Parameters:
        targets (list[TargetSpec]): Tasks whose targets to compose (all DENSE —
            guaranteed by the compatibility guard).
        input_key (str): ``Batch.inputs`` key (image stream) to compose.
        center_ratio (tuple[float, float]): Range for the split center as a
            fraction of height/width; one split is sampled per batch within these
            bounds.
    """

    supported_topologies: frozenset[Topology] = frozenset({Topology.DENSE})

    def __init__(
        self,
        targets: list[TargetSpec],
        input_key: str = IMAGE,
        center_ratio: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        low, high = center_ratio
        if not (0.0 < low <= high < 1.0):
            raise ValueError(f"Mosaic: center_ratio must satisfy 0 < low <= high < 1, got {center_ratio}.")
        self._targets = targets
        self._input_key = input_key
        self._center_ratio = center_ratio

    def __call__(self, batch: Batch) -> Batch:
        height, width = batch.inputs[self._input_key].shape[-2:]
        split_y = self._sample_split(height)
        split_x = self._sample_split(width)

        inputs = {**batch.inputs, self._input_key: self._mosaic(batch.inputs[self._input_key], split_y, split_x)}
        targets = {
            **batch.targets,
            **{spec.key: self._mosaic(batch.targets[spec.key], split_y, split_x) for spec in self._targets},
        }
        return Batch(inputs=inputs, targets=targets, meta=batch.meta)

    @staticmethod
    def _mosaic(x: Tensor, split_y: int, split_x: int) -> Tensor:
        """Swap three quadrants in from rolled batch neighbours (last two dims are H, W).

        Slicing before ``roll`` shifts only the quadrant window across the batch,
        not a full copy of ``x``. Works for both images ``[B, C, H, W]`` and masks
        ``[B, H, W]`` via the trailing-dims ellipsis.
        """
        out = x.clone()
        out[..., :split_y, split_x:] = x[..., :split_y, split_x:].roll(1, 0)  # top-right    ← neighbour 1
        out[..., split_y:, :split_x] = x[..., split_y:, :split_x].roll(2, 0)  # bottom-left  ← neighbour 2
        out[..., split_y:, split_x:] = x[..., split_y:, split_x:].roll(3, 0)  # bottom-right ← neighbour 3
        return out

    def _sample_split(self, size: int) -> int:
        low = max(1, int(self._center_ratio[0] * size))
        high = max(low + 1, int(self._center_ratio[1] * size))
        return int(torch.randint(low, high, (1,)).item())
