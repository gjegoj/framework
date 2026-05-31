"""Collation: stack a list of samples into a batched ``Batch``."""

from __future__ import annotations

import torch

from src.core.entities import Batch, Sample


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack per-sample input/target tensors into batched tensors.

    Parameters:
        samples (list[Sample]): Samples produced by a ``Dataset`` (tensors only).

    Returns:
        Batch: Batched inputs/targets plus per-sample indices in ``meta``.
    """
    inputs = {key: torch.stack([sample.inputs[key] for sample in samples]) for key in samples[0].inputs}
    targets = {key: torch.stack([sample.targets[key] for sample in samples]) for key in samples[0].targets}
    meta = {"index": [sample.meta.get("index") for sample in samples]}
    return Batch(inputs=inputs, targets=targets, meta=meta)
