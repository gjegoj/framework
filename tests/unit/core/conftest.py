"""Values the core tests share: one nested tensor tree and one batch built from it."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.core import Batch, TensorTree


@pytest.fixture
def tree() -> TensorTree:
    """Every container the tree grammar allows, with an absent leaf a tensor operation must leave alone."""
    return {
        "image": torch.zeros(2, 3),
        "text": {"input_ids": torch.ones(2, 4, dtype=torch.long), "mask": [torch.ones(2), torch.zeros(2)]},
        "views": (torch.zeros(1), None),
    }


@pytest.fixture
def batch(tree: TensorTree) -> Batch:
    return Batch(inputs={"x": tree}, targets={"y": torch.tensor([0, 1])}, count=2, metadata={"ids": ["a", "b"]})


def leaves(tree: TensorTree) -> list[Tensor]:
    """Tensors of a tree in traversal order, so two trees can be compared leaf by leaf."""
    if isinstance(tree, Tensor):
        return [tree]
    if isinstance(tree, dict):
        return [leaf for value in tree.values() for leaf in leaves(value)]
    if isinstance(tree, list | tuple):
        return [leaf for value in tree for leaf in leaves(value)]
    return []
