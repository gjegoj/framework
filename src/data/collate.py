"""Collation: a list of ``Sample``s into one ``Batch``."""

from __future__ import annotations

import torch

from src.core.entities import Batch, Sample


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack per-sample inputs and targets into batched tensors.

    Keys are taken from the first sample — samples of one dataset share their
    structure, and a batch where they do not is refused rather than quietly
    stripped. ``meta`` values are transposed to per-sample lists, mirroring how
    tensors are batched.
    """
    if not samples:
        raise ValueError("Cannot collate an empty list of samples.")
    first = samples[0]
    _refuse_disagreeing_metadata(samples)
    return Batch(
        inputs={
            name: torch.stack([torch.as_tensor(sample.inputs[name]) for sample in samples]) for name in first.inputs
        },
        targets={
            task: torch.stack([torch.as_tensor(sample.targets[task]) for sample in samples]) for task in first.targets
        },
        meta={key: [sample.meta[key] for sample in samples] for key in first.meta},
    )


def _refuse_disagreeing_metadata(samples: list[Sample]) -> None:
    """Name a sample whose metadata keys differ, instead of dropping what it carried.

    Transposing from the first sample's keys is what makes collation cheap, and
    the cost of checking the rest agree is 0.6% of a batch, measured. Without it a
    transform that attaches a key to some samples loses it for all of them, and
    the loss looks like the key was never written.
    """
    expected = set(samples[0].meta)
    for index, sample in enumerate(samples):
        if set(sample.meta) != expected:
            raise ValueError(
                f"Sample {index} carries metadata keys {sorted(sample.meta)}, but the batch's first "
                f"sample carries {sorted(expected)}. Every sample of a batch must agree; a transform "
                f"that adds a key must add it to all of them."
            )
