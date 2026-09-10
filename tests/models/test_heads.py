"""A head is any module built at `(in_features, out_features)`: that one contract is what makes it swappable."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.heads import ConvHead, LinearHead
from src.models.registry import head_registry


@pytest.mark.parametrize("name", list(head_registry))
def test_every_registered_head_is_built_from_two_widths_alone(name: str) -> None:
    head = head_registry.get(name)(in_features=6, out_features=2)

    assert isinstance(head, nn.Module)


def test_a_linear_head_projects_a_pooled_vector_onto_the_classes() -> None:
    assert LinearHead(in_features=6, out_features=2)(torch.zeros(4, 6)).shape == (4, 2)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["pointwise", "wider kernel keeps the map size"])
def test_a_conv_head_projects_a_feature_map_and_keeps_its_size(kernel_size: int) -> None:
    head = ConvHead(in_features=6, out_features=2, kernel_size=kernel_size)

    assert head(torch.zeros(4, 6, 5, 5)).shape == (4, 2, 5, 5)
