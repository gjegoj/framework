"""What a batch supervises: the other draw of a picture is the answer, and every other picture is not."""

from __future__ import annotations

import pytest
import torch
from torch.nn.functional import normalize

from src.losses.contrastive import PAIR, InfoNce

SAMPLES, WIDTH = 4, 8


def drawn(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Two draws of every sample, in the order a viewing backbone leaves them: each sample's, adjacent."""
    return torch.stack([first, second], dim=1).flatten(0, 1)


def rows() -> torch.Tensor:
    return torch.arange(SAMPLES)


def test_draws_that_already_point_the_same_way_are_what_this_objective_descends_towards() -> None:
    """The pair is the only right answer in the batch, so agreement is the floor and anything else is above it."""
    directions = normalize(torch.eye(SAMPLES, WIDTH), dim=-1)
    agreed = InfoNce()(drawn(directions, directions), rows())
    apart = InfoNce()(drawn(directions, directions.roll(1, dims=0)), rows())

    assert float(agreed.total) < float(apart.total)


def test_the_pair_is_recovered_from_the_rows_next_to_each_other_and_not_from_any_other_grouping() -> None:
    """A viewing backbone leaves every draw of a sample adjacent, and this objective reads that order.

    Grouped any other way — the first half against the second, say — the matrix's diagonal would pair a
    picture with a different picture's draw, and the run would descend towards telling them apart.
    """
    directions = normalize(torch.eye(SAMPLES, WIDTH), dim=-1)
    adjacent = drawn(directions, directions)

    paired = InfoNce()(adjacent, rows())
    halved = InfoNce()(torch.cat([adjacent[::PAIR], adjacent[1::PAIR]]), rows())

    assert float(paired.total) < float(halved.total)


def test_which_draw_is_read_first_changes_nothing() -> None:
    """Both directions are scored, so neither side of a pair alone is responsible for finding the other.

    Read one way only, the number would depend on which draw the stage happened to stack first — a
    property of the shuffle rather than of the model.
    """
    torch.manual_seed(0)
    first, second = torch.randn(SAMPLES, WIDTH), torch.randn(SAMPLES, WIDTH)

    forward = InfoNce()(drawn(first, second), rows())
    backward = InfoNce()(drawn(second, first), rows())

    assert float(forward.total) == pytest.approx(float(backward.total), abs=1e-6)


def test_the_temperature_is_learned_with_the_run_rather_than_fixed_at_the_declaration() -> None:
    """CLIP learns the scale; keeping it as a parameter is what puts it in the optimizer and the checkpoint."""
    objective = InfoNce()

    objective(drawn(torch.randn(SAMPLES, WIDTH), torch.randn(SAMPLES, WIDTH)), rows()).total.backward()

    assert [name for name, _ in objective.named_parameters()] == ["log_scale"]
    assert objective.log_scale.grad is not None


def test_a_run_that_draws_no_views_is_refused_with_what_it_should_have_declared() -> None:
    """One row per sample means nothing was drawn twice, and there is no pair to compare at all."""
    with pytest.raises(ValueError, match="MultiViewTransform"):
        InfoNce()(torch.randn(SAMPLES, WIDTH), rows())


@pytest.mark.parametrize("temperature", [0.0, -1.0], ids=["zero", "negative"])
def test_a_temperature_that_cannot_divide_is_refused_where_it_is_declared(temperature: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        InfoNce(temperature=temperature)
