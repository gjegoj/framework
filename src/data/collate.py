"""Prepared samples into one batch: tensors stacked along a new first axis, metadata gathered per key."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from src.core import Batch, Sample, TensorTree
from src.data.registry import collator_registry


@collator_registry.register("stack")
class StackCollator:
    """Every sample carries the same tree; each tensor leaf gains a batch axis, an absent leaf stays absent."""

    def __call__(self, samples: Sequence[Sample]) -> Batch:
        if not samples:
            raise ValueError("Cannot collate an empty list of samples.")
        first = samples[0]
        for index, sample in enumerate(samples):
            for part in ("inputs", "targets", "metadata"):
                if set(getattr(sample, part)) != set(getattr(first, part)):
                    odd = sorted(set(getattr(sample, part)) ^ set(getattr(first, part)))
                    raise ValueError(
                        f"Sample {index} disagrees with the first on {part} {odd}; a batch shares one tree."
                    )
        return Batch(
            inputs={name: stack_trees([sample.inputs[name] for sample in samples]) for name in first.inputs},
            targets={name: stack_trees([sample.targets[name] for sample in samples]) for name in first.targets},
            count=len(samples),
            metadata={key: [sample.metadata[key] for sample in samples] for key in first.metadata},
        )


def stack_trees(values: Sequence[object]) -> TensorTree:
    first = values[0]
    if first is None:
        return None
    if isinstance(first, Tensor):
        return torch.stack([torch.as_tensor(value) for value in values])
    if isinstance(first, Mapping):
        return {key: stack_trees([value[key] for value in values]) for key in first}  # type: ignore[index]
    if isinstance(first, list):
        return [stack_trees(items) for items in zip(*values, strict=True)]
    if isinstance(first, tuple):
        return tuple(stack_trees(items) for items in zip(*values, strict=True))
    return torch.stack([torch.as_tensor(value) for value in values])
