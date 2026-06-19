"""Collation: stack a list of samples into a batched ``Batch``."""

from __future__ import annotations

import torch

from src.core.entities import Batch, BatchMeta, Sample


def collate_samples(samples: list[Sample]) -> Batch:
    """Stack per-sample input/target tensors into batched tensors.

    Parameters:
        samples (list[Sample]): Samples produced by a ``Dataset`` (tensors only).

    Returns:
        Batch: Batched inputs/targets plus aggregated provenance ``meta``.
    """
    inputs = {key: torch.stack([sample.inputs[key] for sample in samples]) for key in samples[0].inputs}
    targets = {key: torch.stack([sample.targets[key] for sample in samples]) for key in samples[0].targets}
    return Batch(inputs=inputs, targets=targets, meta=_collate_meta(samples))


def _collate_meta(samples: list[Sample]) -> BatchMeta:
    """Aggregate per-sample ``SampleMeta`` into the batched ``BatchMeta``.

    ``index`` becomes a per-sample list; ``input_sources``/``target_sources`` are
    transposed from per-sample ``{name: path}`` into ``{name: [paths]}`` — so the
    source of sample ``j`` is ``batch.meta["input_sources"][alias][j]``.
    """
    return BatchMeta(
        index=[sample.meta["index"] for sample in samples],
        input_sources=_transpose([sample.meta["input_sources"] for sample in samples]),
        target_sources=_transpose([sample.meta["target_sources"] for sample in samples]),
    )


def _transpose(per_sample: list[dict[str, str]]) -> dict[str, list[str]]:
    """Turn a list of per-sample ``{name: path}`` maps into ``{name: [paths]}``."""
    if not per_sample or not per_sample[0]:
        return {}
    return {name: [entry[name] for entry in per_sample] for name in per_sample[0]}
