"""A head is any module built at `(in_features, out_features)`: that one contract is what makes it swappable."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.core import CLASS_AXIS, Axis
from src.models.base import ShapeAware
from src.models.heads import ConvHead, LinearHead
from src.models.registry import head_registry

WIDTH, CLASSES = 6, 2
FEATURES: dict[tuple[str, ...], tuple[int, ...]] = {
    (Axis.CHANNELS,): (4, WIDTH),
    (Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH): (4, WIDTH, 5, 5),
}
"""A batch of the feature each declared shape stands for: a new shape a head may read needs a row here."""


@pytest.mark.parametrize("name", list(head_registry))
def test_every_registered_head_is_built_from_two_widths_and_reads_what_it_declares(name: str) -> None:
    """The whole head contract: two widths in, and a feature of the shape it says it reads."""
    head = head_registry.get(name)(in_features=WIDTH, out_features=CLASSES)
    reads = tuple(head.reads) if isinstance(head, ShapeAware) else (Axis.CHANNELS,)
    assert reads in FEATURES, f"{name!r} reads {reads}, which has no specimen; add one to FEATURES."

    produced = head(torch.zeros(*FEATURES[reads]))

    assert isinstance(head, nn.Module) and produced.shape[CLASS_AXIS] == CLASSES


def test_a_linear_head_projects_a_pooled_vector_onto_the_classes() -> None:
    assert LinearHead(in_features=6, out_features=2)(torch.zeros(4, 6)).shape == (4, 2)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["pointwise", "wider kernel keeps the map size"])
def test_a_conv_head_projects_a_feature_map_and_keeps_its_size(kernel_size: int) -> None:
    head = ConvHead(in_features=6, out_features=2, kernel_size=kernel_size)

    assert head(torch.zeros(4, 6, 5, 5)).shape == (4, 2, 5, 5)
