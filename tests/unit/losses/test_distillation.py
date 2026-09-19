"""Learning from a second network: what the objective compares, and what a bounded answer needs first."""

from __future__ import annotations

import pytest
import torch

from src.core import Representation
from src.losses.distillation import KullbackLeibler

TEACHER = torch.tensor([[0.0027, 0.4077, 0.0303, 0.0326, 0.9063, 0.1389, 0.0157, 0.4251]])
"""One real answer of an eight-class ``cosine`` head trained under ``arcface``, cosines as they ship."""

STUDENT = torch.tensor([[0.1100, 0.3200, 0.0800, 0.0500, 0.7400, 0.2000, 0.0400, 0.3900]])
"""An answer beside it that agrees about the class and not about the rest — which is what is learned."""


def divergence_of(temperature: float = 4.0, scale: float | None = None) -> float:
    """How far the student above is from the teacher above, as this objective scores it."""
    return float(KullbackLeibler(temperature=temperature, scale=scale)(STUDENT, TEACHER).total)


@pytest.mark.parametrize(
    ("scale", "reads"),
    [
        pytest.param(None, Representation.PROJECTED, id="a head whose answer is already a logit"),
        pytest.param(16.0, Representation.COSINES, id="a head answering in angles"),
        pytest.param(1.0, Representation.COSINES, id="a conversion by one is still a conversion"),
    ],
)
def test_what_this_objective_reads_is_settled_by_whether_a_scale_was_declared(
    scale: float | None, reads: Representation
) -> None:
    """A scale turns cosines into logits and nothing else needs one, so declaring it declares the reading.

    Written this way rather than as a third knob repeating what the head already says: the pair is
    checked where a run is assembled, so a scale over a projection and a projection under no scale are
    both refused by name there. A knob restating the head would be a second statement free to disagree.
    """
    assert KullbackLeibler(scale=scale).reads == reads


@pytest.mark.parametrize(
    ("temperature", "scale", "divergence"),
    [
        pytest.param(4.0, None, 0.0036518, id="an answer already a logit, softened as the default softens"),
        pytest.param(4.0, 1.0, 0.0036518, id="the same arithmetic, declared as a conversion by one"),
        pytest.param(1.0, 4.0, 0.0528893, id="angles made logits, unsoftened"),
        pytest.param(2.0, 8.0, 0.2115573, id="the same comparison, softened twice as far"),
        pytest.param(4.0, 16.0, 0.8462291, id="the same comparison again, softened four times as far"),
    ],
)
def test_what_is_compared_is_the_ratio_and_what_it_weighs_is_the_square_of_the_temperature(
    temperature: float, scale: float | None, divergence: float
) -> None:
    """The two knobs do two jobs, and the table is where both are visible at once.

    ``softmax(scale * cos / temperature)`` holds one degree of freedom, so the last three rows soften
    the two answers identically — measured, all three put the teacher's winning class at 0.6924 — and
    differ only by the factor that keeps a share of the objective meaning the same at any temperature:
    1, 4, 16, which is the temperature squared. That is also why a bounded answer gets a scale rather
    than a temperature below one: the knob that would sharpen it is the knob that throws it away.

    The first two rows are the same arithmetic under two readings, and they have to stay the same
    arithmetic: a conversion by one converts nothing, and what it declares is which space the answer is
    in, which only the head it is held against can settle.
    """
    assert divergence_of(temperature=temperature, scale=scale) == pytest.approx(divergence, rel=1e-4)


@pytest.mark.parametrize(
    ("declared", "refused"),
    [
        pytest.param({"temperature": 0.0}, "temperature", id="a temperature that divides by nothing"),
        pytest.param({"temperature": -1.0}, "temperature", id="a temperature below zero"),
        pytest.param({"scale": 0.0}, "scale", id="a scale that flattens every answer to one"),
        pytest.param({"scale": -2.0}, "scale", id="a scale that turns the answer inside out"),
    ],
)
def test_a_declaration_that_would_compare_nothing_is_refused_where_it_is_written(
    declared: dict[str, float], refused: str
) -> None:
    """Both numbers divide or multiply an answer before it is a distribution, so both are positive."""
    with pytest.raises(ValueError, match=refused):
        KullbackLeibler(**declared)
