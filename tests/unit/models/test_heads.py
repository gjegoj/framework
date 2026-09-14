"""A head is any module built at `(in_features, out_features)`: that one contract is what makes it swappable."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.core import FEATURE_AXIS, Axis
from src.models.base import ShapeAware
from src.models.heads import ConvHead, CosineHead, ExpandedHead, LinearHead, StackedHeads
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
    reads = tuple(head.reads_axes) if isinstance(head, ShapeAware) else (Axis.CHANNELS,)
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


class TestExpandedHead:
    """A class space that grew: what was learned keeps its indices, what is new is appended after them."""

    def test_the_classes_a_run_already_had_keep_the_indices_they_had(self) -> None:
        """The declared vocabulary pins index to name, so a grown run reads a kept file's rows as before."""
        grown = ExpandedHead(base=LinearHead(WIDTH, CLASSES), novel=LinearHead(WIDTH, 3))
        features = torch.randn(4, WIDTH)

        answered = grown(features)

        assert answered.shape == (4, CLASSES + 3)
        assert torch.allclose(answered[:, :CLASSES], grown.base(features))

    def test_a_dense_output_grows_on_the_same_axis_a_flat_one_does(self) -> None:
        """`[B, C, H, W]` and `[B, C]` name their classes on the same axis, and one rule covers both."""
        grown = ExpandedHead(base=ConvHead(WIDTH, CLASSES), novel=ConvHead(WIDTH, 1))

        answered = grown(torch.randn(4, WIDTH, 5, 5))

        assert answered.shape == (4, CLASSES + 1, 5, 5)

    def test_what_was_learned_and_what_is_new_are_addressed_apart(self) -> None:
        """`freeze`, the optimizer and the averaging callback all walk paths; two submodules are that contract.

        One wider matrix would put both under one tensor, and ``requires_grad`` lives on whole tensors —
        a run holding the carried rows still would hold the new ones still with them.
        """
        grown = ExpandedHead(base=LinearHead(WIDTH, CLASSES), novel=LinearHead(WIDTH, 3))

        assert {name for name, _ in grown.named_children()} == {"base", "novel"}
        assert "base.projection.weight" in grown.state_dict()


class TestStacked:
    """Several streams, one head each, their answers folded into the batch the way a sample's draws are."""

    def test_every_stream_is_read_by_a_head_built_for_its_own_width(self) -> None:
        """What arrives is one answer per stream per sample, which is what a batch of them holds."""
        stacked = StackedHeads({"first": nn.Linear(2, CLASSES), "second": nn.Linear(4, CLASSES)})

        answered = stacked(torch.zeros(3, 2), torch.zeros(3, 4))

        assert tuple(answered.shape) == (3 * 2, CLASSES)

    def test_the_answers_of_one_sample_stay_next_to_each_other(self) -> None:
        """The order an objective comparing a pair recovers them by; it is the contract, not an accident."""
        stacked = StackedHeads({"first": nn.Identity(), "second": nn.Identity()})

        answered = stacked(torch.tensor([[1.0], [2.0]]), torch.tensor([[10.0], [20.0]]))

        assert answered.flatten().tolist() == [1.0, 10.0, 2.0, 20.0]
