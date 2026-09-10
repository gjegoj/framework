"""A collator stacks same-shaped trees into one batch and carries every sample's metadata along."""

from __future__ import annotations

import pytest
import torch

from src.core import Sample, require_tensor
from src.data.collate import StackCollator


@pytest.fixture
def samples() -> list[Sample]:
    return [
        Sample(
            inputs={"image": torch.full((3, 2, 2), float(i)), "text": {"ids": torch.tensor([i, i])}},
            targets={"label": torch.tensor(i), "boxes": None},
            metadata={"row": i, "cells": {"path": f"{i}.png"}},
        )
        for i in range(3)
    ]


def test_stacks_every_tensor_leaf_and_keeps_the_tree(samples: list[Sample]) -> None:
    batch = StackCollator()(samples)

    text = batch.inputs["text"]

    assert len(batch) == 3
    assert require_tensor(batch.inputs["image"], name="image").shape == (3, 3, 2, 2)
    assert isinstance(text, dict) and require_tensor(text["ids"], name="ids").tolist() == [[0, 0], [1, 1], [2, 2]]
    assert require_tensor(batch.targets["label"], name="label").tolist() == [0, 1, 2]
    assert batch.targets["boxes"] is None


def test_metadata_becomes_one_list_per_key(samples: list[Sample]) -> None:
    batch = StackCollator()(samples)

    assert batch.metadata == {"row": [0, 1, 2], "cells": [{"path": "0.png"}, {"path": "1.png"}, {"path": "2.png"}]}


def test_refuses_an_empty_batch_or_samples_that_disagree(samples: list[Sample]) -> None:
    with pytest.raises(ValueError):
        StackCollator()([])
    odd = Sample(inputs={"image": torch.zeros(3, 2, 2)}, targets={}, metadata={})
    with pytest.raises(ValueError, match="text"):
        StackCollator()([*samples, odd])
