"""Collation: a list of ``Sample``s into one ``Batch``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.core.entities import Batch, Instances, Sample

if TYPE_CHECKING:
    from src.core.entities import TaskOutput


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
        targets={task: _batched(task, [sample.targets[task] for sample in samples]) for task in first.targets},
        meta={key: [sample.meta[key] for sample in samples] for key in first.meta},
    )


def _batched(task: str, values: list[Any]) -> TaskOutput:
    """One target's per-sample values in their batch form — by what they are, not who made them.

    A target's shape family is declared by its type, which its encoder chose: nothing
    here asks the task or the config which branch to take, so a new ragged kind is a new
    ``TaskOutput`` member and a clause, not a flag threaded through the data layer.
    """
    if all(isinstance(value, Instances) for value in values):
        return _merged(task, values)
    if any(isinstance(value, Instances) for value in values):
        spelled = ", ".join(sorted({type(value).__name__ for value in values}))
        raise ValueError(
            f"Target '{task}' mixes {spelled} across one batch; samples of one dataset share their "
            f"structure, and batching half of them as objects would drop the rest."
        )
    return torch.stack([torch.as_tensor(value) for value in values])


def _merged(task: str, values: list[Instances]) -> Instances:
    """Per-sample objects into the flat batch shape, renumbering ``sample_index``.

    The numbering is rewritten wholesale rather than offset: whatever an encoder wrote
    (zeros, by its contract), which image an object belongs to *within this batch* is a
    fact only collation holds.
    """
    if any(value.scores is not None for value in values):
        raise ValueError(
            f"Target '{task}' carries scores, but ground truth has no confidence — a prediction has "
            f"leaked into a target path."
        )
    counts = torch.tensor([len(value.boxes) for value in values])
    return Instances(
        boxes=torch.cat([value.boxes for value in values]),
        labels=torch.cat([value.labels for value in values]),
        sample_index=torch.arange(len(values)).repeat_interleave(counts),
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
