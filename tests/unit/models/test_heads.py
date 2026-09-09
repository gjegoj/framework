"""Built-in heads: projections over one stream, refusing a pyramid by name."""

from __future__ import annotations

import pytest
import torch

from src.core.ports import one_stream
from src.models import ConvHead, LinearHead


def test_linear_head_projects_to_out_features() -> None:
    head = LinearHead(in_features=4, out_features=2)

    assert head(torch.zeros(3, 4)).shape == (3, 2)


@pytest.mark.parametrize("kernel", [{}, {"kernel_size": 3}], ids=["default", "wider"])
def test_conv_head_projects_channels_and_keeps_spatial_dims(kernel: dict[str, int]) -> None:
    """A dense head predicts one value per pixel, so same-padding is not optional."""
    head = ConvHead(in_features=16, out_features=3, **kernel)

    assert head(torch.zeros(2, 16, 8, 8)).shape == (2, 3, 8, 8)


def test_a_single_stream_head_refuses_a_pyramid_by_name() -> None:
    """A head over one stream handed several has been declared on the wrong topology."""
    pyramid = {"p3": torch.zeros(1, 4, 8, 8), "p4": torch.zeros(1, 8, 4, 4)}

    with pytest.raises(TypeError, match="LinearHead reads one stream.*p3, p4"):
        LinearHead(4, 2)(pyramid)


def test_one_stream_passes_a_tensor_through() -> None:
    features = torch.zeros(2, 4)

    assert one_stream(features, head="LinearHead") is features
