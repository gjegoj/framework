"""Where a computed value goes: a number to the log, a picture to whoever can draw one.

The routing is the whole of this module, so these are tests about geometry rather than about metrics —
what a scalar, a per-class vector and a matrix each become, and what is said about a value that is none
of the three.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Matrix, Stage
from src.tracking import MetricKey, report

KEY = MetricKey(Stage.VAL, "f1", task="species")
CLASSES = {0: "cat", 1: "dog"}


class Written(dict[str, float]):
    """A stand-in for ``LightningModule.log``: every scalar it was handed, by key."""

    def __call__(self, key: str, value: Any) -> None:
        self[key] = float(value)


class Chart:
    """A tracker that can draw a matrix, and remembers what it was asked to draw."""

    def __init__(self) -> None:
        self.drawn: list[tuple[str, Matrix, int]] = []

    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        self.drawn.append((title, matrix, iteration))


class Plain:
    """A tracker that takes scalars and nothing else — a CSV file, in essence."""


@pytest.fixture
def written() -> Written:
    return Written()


@pytest.fixture
def matrix() -> Matrix:
    return Matrix(torch.eye(2), xaxis="Predicted", yaxis="True")


def send(value: object, *, written: Written, trackers: tuple[object, ...] = (), classes: Any = CLASSES) -> None:
    report(KEY, value, scalar_log=written, trackers=trackers, step=3, classes=classes)


class TestNumbers:
    def test_a_single_number_is_written_under_the_key_itself(self, written: Written) -> None:
        send(torch.tensor(0.75), written=written)

        assert written == {"val/species/f1": 0.75}

    def test_a_reading_per_class_becomes_a_line_per_class_and_the_mean_they_read_against(
        self, written: Written
    ) -> None:
        """One graph with a line each: the leaves share the family `val/species/f1`."""
        send(torch.tensor([0.5, 1.0]), written=written)

        assert written == {"val/species/f1/cat": 0.5, "val/species/f1/dog": 1.0, "val/species/f1/mean": 0.75}

    def test_a_reading_the_vocabulary_cannot_cover_is_written_by_position(self, written: Written) -> None:
        """A samplewise reading has one value per row, and naming those after classes would be a lie."""
        send(torch.tensor([1.0, 2.0, 3.0]), written=written)

        assert set(written) == {f"val/species/f1/class{index}" for index in range(3)} | {"val/species/f1/mean"}

    def test_a_reading_no_vocabulary_names_is_written_by_position(self, written: Written) -> None:
        """A per-output regression reading has no classes at all, and is still worth reporting."""
        send(torch.tensor([1.0, 2.0]), written=written, classes=None)

        assert set(written) == {"val/species/f1/class0", "val/species/f1/class1", "val/species/f1/mean"}


class TestPictures:
    def test_a_picture_reaches_every_tracker_that_can_draw_one(self, written: Written, matrix: Matrix) -> None:
        charts = (Chart(), Chart())

        send(matrix, written=written, trackers=charts)

        assert [[(title, step) for title, _, step in chart.drawn] for chart in charts] == [[("val/species/f1", 3)]] * 2
        assert all(chart.drawn[0][1].value is matrix.value for chart in charts), "the reading itself is untouched"
        assert written == {}, "a picture is not a number, and the log would refuse it"

    def test_a_tracker_that_draws_nothing_is_passed_over_rather_than_asked(
        self, written: Written, matrix: Matrix
    ) -> None:
        """Structural: a backend qualifies by having the method, and the run keeps its other trackers."""
        chart = Chart()

        send(matrix, written=written, trackers=(Plain(), chart))

        assert len(chart.drawn) == 1

    def test_a_picture_nothing_in_the_run_can_draw_is_said_out_loud_once(
        self, written: Written, matrix: Matrix
    ) -> None:
        """Named without its stage: the answer is the same in train, val and test, so it is said once."""
        with pytest.warns(UserWarning, match=r"^species/f1 is a picture"):
            send(matrix, written=written, trackers=(Plain(),))

    def test_a_picture_is_named_by_the_vocabulary_the_task_declared(self, written: Written, matrix: Matrix) -> None:
        """The metric counts; only the task knows what the classes are called, and the axes read from it."""
        chart = Chart()

        send(matrix, written=written, trackers=(chart,))

        assert chart.drawn[0][1].labels == ("cat", "dog")

    def test_a_picture_whose_task_named_nothing_is_drawn_as_it_came(self, written: Written, matrix: Matrix) -> None:
        chart = Chart()

        send(matrix, written=written, trackers=(chart,), classes=None)

        assert chart.drawn[0][1].labels is None

    def test_a_run_that_records_nothing_says_nothing(
        self, written: Written, matrix: Matrix, recwarn: pytest.WarningsRecorder
    ) -> None:
        """`tracker: none` is a declaration, not an oversight: there is nowhere to draw and no complaint."""
        send(matrix, written=written)

        assert list(recwarn) == []


class TestUnknownShapes:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(torch.zeros(2, 2), id="a bare 2-D tensor"),
            pytest.param((torch.zeros(2), torch.zeros(2)), id="a pair of curves"),
        ],
    )
    def test_a_value_of_a_shape_nothing_was_told_how_to_draw_is_named_rather_than_dropped(
        self, written: Written, value: object
    ) -> None:
        with pytest.warns(UserWarning, match="val/species/f1"):
            send(value, written=written, trackers=(Chart(),))

        assert written == {}
