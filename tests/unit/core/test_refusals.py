"""Saying where in a declaration a refusal came from, without losing the refusal."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from src.core import naming


class Declared(BaseModel):
    """Any validated section, standing in for the ones a builder validates inside a position."""

    width: int


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(LookupError("Unknown loss 'focl'"), id="a name nothing implements"),
        pytest.param(ValueError("'focal' takes no gama"), id="a knob no constructor takes"),
        pytest.param(TypeError("Linear() takes 2 positional arguments"), id="a call that could not be made"),
    ],
)
def test_a_refusal_raised_inside_a_position_arrives_with_that_position_in_front_of_it(
    raised: Exception,
) -> None:
    """The position is the one thing the refusal cannot know and the reader needs: which line to change."""
    with pytest.raises(type(raised), match=re.escape(f"tasks.mask.loss: {raised}")), naming("tasks.mask.loss"):
        raise raised


def test_a_refusal_a_library_raised_keeps_its_words_and_stays_the_kind_it_was() -> None:
    """Rebuilding a foreign exception as its own class is what loses it.

    Measured on pydantic 2.13.4: ``ValidationError`` is a ``ValueError`` whose constructor takes line
    errors rather than a message, so re-raising one as itself answered with
    ``ValidationError.__new__() missing 1 required positional argument: 'line_errors'`` — a refusal
    naming the declaration and the fix, replaced by the internals of the library that raised it.
    """
    with pytest.raises(ValueError, match=r"(?s)tasks\.mask\.head: .*width"), naming("tasks.mask.head"):
        Declared.model_validate({"width": "wide"})


def test_what_a_position_says_nothing_about_travels_on_untouched() -> None:
    """Only a refusal about a declaration is named; a run that broke some other way is not this one's."""
    with pytest.raises(KeyboardInterrupt), naming("tasks.mask.loss"):
        raise KeyboardInterrupt


def test_the_refusal_that_arrived_is_kept_as_the_cause() -> None:
    """Named rather than replaced: a traceback still ends at whoever refused, and where."""
    raised = LookupError("Unknown loss 'focl'")

    with pytest.raises(LookupError) as refusal, naming("tasks.mask.loss"):
        raise raised

    assert refusal.value.__cause__ is raised


def test_a_library_refusal_is_still_one_a_caller_can_tell_apart() -> None:
    """A pydantic refusal is a ValueError, and a caller separating a bad declaration from an unknown
    name catches it as one; what cannot survive is the subclass, whose constructor is its own."""
    with pytest.raises(ValidationError):
        Declared.model_validate({"width": "wide"})
    with pytest.raises(ValueError), naming("tasks.mask.head"):
        Declared.model_validate({"width": "wide"})
