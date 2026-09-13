"""A head is any module built at `(in_features, out_features)`: that one contract is what makes it swappable."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.core import FEATURE_AXIS, Axis
from src.models.base import ShapeAware
from src.models.heads import ConvHead, CosineHead, LinearHead
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

    assert isinstance(head, nn.Module) and produced.shape[FEATURE_AXIS] == CLASSES


def test_a_linear_head_projects_a_pooled_vector_onto_the_classes() -> None:
    assert LinearHead(in_features=6, out_features=2)(torch.zeros(4, 6)).shape == (4, 2)


class TestCosine:
    """The arrangement that keeps the prototypes in the network, so what a run ships classifies."""

    def test_it_answers_with_angles_so_how_loud_a_feature_is_cannot_change_the_ranking(self) -> None:
        """A cosine is the whole point: an angular margin has nothing to add a margin to otherwise."""
        head = CosineHead(in_features=6, out_features=2)
        features = torch.rand(4, 6)

        quiet, loud = head(features), head(features * 100)

        assert torch.allclose(quiet, loud, atol=1e-5)
        assert bool(quiet.abs().max() <= 1.0 + 1e-6)

    def test_the_prototypes_it_compares_against_travel_with_the_network(self) -> None:
        """That is what this arrangement is for: the exported artifact answers about the classes itself."""
        head = CosineHead(in_features=6, out_features=3)

        assert head.prototypes.shape == (3, 6)
        assert "prototypes" in head.state_dict()

    def test_a_width_that_is_no_width_is_refused_rather_than_falling_back_to_the_stream(self) -> None:
        """Zero would build, normalize to nan, and train a run that reports numbers the whole way."""
        with pytest.raises(ValueError, match="embedding_dim"):
            CosineHead(in_features=6, out_features=3, embedding_dim=0)

    def test_a_declared_width_is_projected_down_to_before_the_angles_are_taken(self) -> None:
        """A wide backbone against a narrow embedding: the prototypes follow the width, not the stream."""
        head = CosineHead(in_features=6, out_features=3, embedding_dim=4)

        assert head.prototypes.shape == (3, 4) and head(torch.rand(2, 6)).shape == (2, 3)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["pointwise", "wider kernel keeps the map size"])
def test_a_conv_head_projects_a_feature_map_and_keeps_its_size(kernel_size: int) -> None:
    head = ConvHead(in_features=6, out_features=2, kernel_size=kernel_size)

    assert head(torch.zeros(4, 6, 5, 5)).shape == (4, 2, 5, 5)
