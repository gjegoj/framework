"""Turning one sample input into stacked views for a shared encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core.entities import Sample
from src.core.taxonomy import Modality

if TYPE_CHECKING:
    from src.core.ports import SampleTransform


class MultiViewTransform:
    """Stacks N (optionally augmented) copies of one input: ``[...]`` to ``[N, ...]``.

    The base transform runs per view on a throwaway copy of the sample, so random
    augmentations produce distinct views; without a base the views are identical. Only the
    named input is stacked; other inputs and all targets reach the batch unchanged. Pairs
    with ``MultiViewBackbone``, which folds the view axis back into the batch.

    Parameters:
        views (int): Number of views, at least two.
        base (SampleTransform | None): Augmentation applied per view.
        input_name (str): Which input to expand.
    """

    def __init__(
        self,
        views: int,
        base: SampleTransform | None = None,
        input_name: str = Modality.IMAGE,
    ) -> None:
        if views < 2:
            raise ValueError(f"MultiViewTransform needs at least two views, got {views}.")
        self._views = views
        self._base = base
        self._input_name = input_name

    def __call__(self, sample: Sample) -> Sample:
        variants = []
        for _ in range(self._views):
            # Each view augments a throwaway copy: a base transform that also touches
            # targets (masks) must not leave one view's target behind for the next.
            candidate = Sample(
                inputs=dict(sample.inputs),
                targets=dict(sample.targets),
                auxiliary_inputs=dict(sample.auxiliary_inputs),
                meta=sample.meta,
            )
            if self._base is not None:
                candidate = self._base(candidate)
            variants.append(torch.as_tensor(candidate.inputs[self._input_name]))
        sample.inputs[self._input_name] = torch.stack(variants)
        return sample
