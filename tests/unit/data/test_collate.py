"""Collation: a target batches by its shape family, declared by its type."""

from __future__ import annotations

import pytest
import torch

from src.core import Instances, Sample
from src.data.collate import collate_samples
from tests.support.narrowing import tensor


def objects(count: int, label: int = 0) -> Instances:
    """One sample's objects, as its encoder writes them: numbering left to collation."""
    return Instances(
        boxes=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
        labels=torch.full((count,), label, dtype=torch.int64),
        sample_index=torch.zeros(count, dtype=torch.int64),
    )


def sample_of(instances: Instances) -> Sample:
    return Sample(inputs={"image": torch.zeros(3, 4, 4)}, targets={"objects": instances})


def test_instances_concatenate_and_the_index_numbers_the_samples() -> None:
    """Whatever an encoder wrote per sample, position in this batch is the collate's fact."""
    batch = collate_samples([sample_of(objects(2, label=0)), sample_of(objects(0)), sample_of(objects(1, label=3))])
    merged = batch.targets["objects"]

    assert isinstance(merged, Instances)
    assert merged.sample_index.tolist() == [0, 0, 2]
    assert merged.labels.tolist() == [0, 0, 3]
    assert merged.boxes.shape == (3, 4)


def test_a_negative_sample_is_a_zero_row_slice_of_the_batch() -> None:
    """An image with nothing in it still occupies its position — ``of`` finds it empty."""
    batch = collate_samples([sample_of(objects(1)), sample_of(objects(0))])
    merged = batch.targets["objects"]

    assert isinstance(merged, Instances)
    assert len(merged.of(1).boxes) == 0
    assert len(merged.of(0).boxes) == 1


def test_tensor_targets_still_stack_beside_ragged_ones() -> None:
    """The multitask batch: one task's objects, another's label, one collate."""
    samples = [
        Sample(inputs={"image": torch.zeros(3, 4, 4)}, targets={"objects": objects(1), "label": torch.tensor(2)}),
        Sample(inputs={"image": torch.zeros(3, 4, 4)}, targets={"objects": objects(2), "label": torch.tensor(0)}),
    ]

    batch = collate_samples(samples)

    assert tensor(batch.targets["label"]).tolist() == [2, 0]
    assert isinstance(batch.targets["objects"], Instances)


def test_a_batch_mixing_shape_families_for_one_task_is_refused() -> None:
    """Samples of one dataset share their structure; guessing which half to believe would not."""
    mixed = [
        sample_of(objects(1)),
        Sample(inputs={"image": torch.zeros(3, 4, 4)}, targets={"objects": torch.tensor(1)}),
    ]

    with pytest.raises(ValueError, match="objects"):
        collate_samples(mixed)


def test_ground_truth_arriving_with_scores_is_refused() -> None:
    """A prediction leaked into a target path is a bug worth naming, not averaging away."""
    scored = Instances(
        boxes=torch.zeros(1, 4),
        labels=torch.zeros(1, dtype=torch.int64),
        sample_index=torch.zeros(1, dtype=torch.int64),
        scores=torch.ones(1),
    )

    with pytest.raises(ValueError, match="scores"):
        collate_samples([sample_of(scored)])
