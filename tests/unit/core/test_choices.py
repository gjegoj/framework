"""``one_of``: the runtime half of a ``Literal``, which the interpreter throws away."""

from __future__ import annotations

from typing import Literal

import pytest

from src.core import one_of

type Pooling = Literal["cls", "mean"]


def test_an_accepted_value_passes_through_unchanged() -> None:
    """The guard sits in an assignment, so it has to hand back what it was given."""
    assert one_of("mean", Pooling) == "mean"


def test_a_misspelt_value_is_refused_with_the_options_beside_it() -> None:
    """A typo silently selecting another branch is the failure this exists to prevent."""
    with pytest.raises(ValueError, match="Pooling must be one of cls, mean, got 'men'"):
        one_of("men", Pooling)


def test_the_alias_supplies_both_the_options_and_the_name() -> None:
    """Nothing is repeated at a call site: whoever declares the set has already named it."""
    type Reduction = Literal["mean", "sum", "none"]

    with pytest.raises(ValueError, match="Reduction must be one of mean, sum, none"):
        one_of("meen", Reduction)
