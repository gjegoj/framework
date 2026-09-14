"""Views are folded into the batch to be encoded: one set of weights sees every draw, in a known order."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.core import Stream
from src.models.backbones.multiview import MultiViewBackbone
from tests.unit.models.conftest import POOLED_WIDTH, SIDE, Encoder

VIEWS, BATCH = 2, 2


def drawn() -> dict[str, torch.Tensor]:
    """Two samples of two draws each, where the draws of one sample are alike and the samples are not."""
    pictures = torch.stack([torch.full((VIEWS, 3, SIDE, SIDE), float(sample)) for sample in range(BATCH)])
    return {"image": pictures}


def test_every_draw_is_encoded_and_the_draws_of_one_sample_stay_next_to_each_other() -> None:
    """The objective reading them recovers the pairs from this order alone, so the order is the contract."""
    encoded = MultiViewBackbone(Encoder())(drawn())[Stream.POOLED]

    assert tuple(encoded.shape) == (BATCH * VIEWS, POOLED_WIDTH)
    assert torch.equal(encoded[0], encoded[1]) and torch.equal(encoded[2], encoded[3])
    assert not torch.equal(encoded[0], encoded[2])


def test_what_a_head_is_sized_from_is_what_one_draw_encodes_to() -> None:
    """Views ride with the batch: no declared shape carries them, so a head is built as it always was."""
    assert MultiViewBackbone(Encoder()).feature_shapes == Encoder().feature_shapes


def test_a_network_that_publishes_no_streams_is_refused_where_it_is_declared() -> None:
    """This one answers with whatever it wraps, so what it wraps has to be something a head can read."""
    with pytest.raises(TypeError, match="Backbone"):
        MultiViewBackbone(nn.Linear(2, 2))  # type: ignore[arg-type]
