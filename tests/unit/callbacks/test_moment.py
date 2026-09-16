"""Where in a run something begins or ends: one grammar, declared the same way by every callback."""

from __future__ import annotations

import pytest

from src.callbacks.moment import Moment
from src.training import FitProfile

RUN = FitProfile(total_steps=100, epochs=10)


@pytest.mark.parametrize(
    ("declared", "epoch"),
    [
        pytest.param(0.34, 4, id="a share lands on the first whole epoch past it"),
        pytest.param(0.25, 3, id="a quarter of ten is three, not two"),
        pytest.param(0.05, 1, id="a share smaller than one epoch is still one"),
        pytest.param(3.0, 3, id="a whole number above one is that epoch itself"),
    ],
)
def test_a_moment_counted_in_epochs(declared: float, epoch: int) -> None:
    """Rounded up, because a share names a point and the boundary is the first whole epoch at or past it.

    Measured before choosing: Python's ``round`` is banker's, so ``round(2.5) == 2`` — a quarter of a
    ten-epoch run would come out one epoch short of its own share while a third came out past it.
    """
    assert Moment(declared, knob="until").in_epochs(RUN).epoch == epoch


@pytest.mark.parametrize(
    ("declared", "step"),
    [
        pytest.param(0.34, 34, id="a share keeps the resolution of a step"),
        pytest.param(3.0, 30, id="a whole number is that epoch's first step"),
    ],
)
def test_a_moment_counted_in_steps(declared: float, step: int) -> None:
    """Read against the run's steps, so a step-wise callback keeps the resolution it counts in."""
    assert Moment(declared, knob="until").in_steps(RUN).step == step


@pytest.mark.parametrize(
    ("moment", "said"),
    [
        pytest.param(Moment(0.34, knob="until").in_epochs(RUN), "epoch 4 (step 40)", id="counted in epochs"),
        pytest.param(Moment(0.34, knob="after", opens=True).in_steps(RUN), "epoch 3 (step 34)", id="counted in steps"),
        pytest.param(Moment(0.0, knob="after", opens=True).in_steps(RUN), "epoch 0 (step 0)", id="from the start"),
    ],
)
def test_a_boundary_is_said_in_both_of_the_units_a_reader_can_act_on(moment: object, said: str) -> None:
    """A config declares a share, a callback acts on an epoch, a tracker counts in steps."""
    assert str(moment) == said


@pytest.mark.parametrize(
    ("knob", "opens", "boundary"),
    [
        pytest.param("until", False, "epoch 10 (step 100)", id="a hold nobody bounded lasts the run"),
        pytest.param("after", True, "epoch 0 (step 0)", id="a wait nobody declared is none"),
    ],
)
def test_a_knob_that_names_no_moment_is_the_end_of_the_run_or_its_beginning(
    knob: str, opens: bool, boundary: str
) -> None:
    """The whole run is the absence of the knob rather than a number that happens to mean it.

    Written as a number it would be 1, which is the one value the two readings disagree about — and
    both units answer the same here, because the end of the run is the end of it counted either way.
    """
    moment = Moment(None, knob=knob, opens=opens)

    assert str(moment.in_epochs(RUN)) == boundary and str(moment.in_steps(RUN)) == boundary


@pytest.mark.parametrize("declared", [1, 1.0], ids=["written whole", "written as a share"])
@pytest.mark.parametrize(
    ("knob", "opens"), [("until", False), ("after", True)], ids=["a knob that closes", "one that opens"]
)
def test_the_one_number_the_two_readings_disagree_about_is_refused_rather_than_read_as_either(
    declared: float, knob: str, opens: bool
) -> None:
    """Every other value reads the same both ways or belongs to one reading alone: 0 is where nothing
    happens on either count, 0.3 is no epoch, 3 is no share. 1 is the whole run *and* its first epoch,
    and a run that meant the second held its backbone for all of the first — under a log line that
    said so, in a message that made the declaration look honoured.
    """
    with pytest.raises(ValueError, match="whole run"):
        Moment(declared, knob=knob, opens=opens)


@pytest.mark.parametrize(
    ("declared", "opens"),
    [
        pytest.param(0.0, False, id="an end at the start: nothing would ever happen"),
        pytest.param(1.5, False, id="neither a share nor a whole epoch"),
        pytest.param(-0.5, False, id="before the run"),
    ],
)
def test_a_moment_outside_the_grammar_is_refused_where_it_was_declared(declared: float, opens: bool) -> None:
    """Refused at construction, naming the knob as the config spells it, so it dies before the run does."""
    with pytest.raises(ValueError, match=r"until|after"):
        Moment(declared, knob="after" if opens else "until", opens=opens)
