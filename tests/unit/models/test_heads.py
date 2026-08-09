"""Built-in heads: linear projection and the identity pass-through."""

from __future__ import annotations

import pytest
import torch

from src.models import ConvHead, IdentityHead, LinearHead


def test_linear_head_projects_to_out_features() -> None:
    head = LinearHead(in_features=4, out_features=2)

    assert head(torch.zeros(3, 4)).shape == (3, 2)


@pytest.mark.parametrize("kernel", [{}, {"kernel_size": 3}], ids=["default", "wider"])
def test_conv_head_projects_channels_and_keeps_spatial_dims(kernel: dict[str, int]) -> None:
    """A dense head predicts one value per pixel, so same-padding is not optional."""
    head = ConvHead(in_features=16, out_features=3, **kernel)

    assert head(torch.zeros(2, 16, 8, 8)).shape == (2, 3, 8, 8)


def test_identity_head_passes_the_stream_through() -> None:
    stream = torch.randn(3, 5)

    assert torch.equal(IdentityHead()(stream), stream)
