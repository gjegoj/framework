"""The heads registry: built-in head_registry are reachable by config-friendly names."""

from __future__ import annotations

import torch

from src.models import LinearHead
from src.models.registry import head_registry


def test_built_in_heads_are_registered() -> None:
    assert set(head_registry) == {"linear", "identity", "conv", "cosine"}


def test_create_builds_a_sized_head() -> None:
    head = head_registry.create("linear", in_features=4, out_features=2)

    assert isinstance(head, LinearHead)
    assert head(torch.zeros(3, 4)).shape == (3, 2)
