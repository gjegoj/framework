"""One grammar for every reported value: ``stage/[task/]name``, read and written in one place."""

from __future__ import annotations

import pytest

from src.core import Stage
from src.tracking import MetricKey


@pytest.mark.parametrize(
    ("key", "text"),
    [
        pytest.param(MetricKey(Stage.TRAIN, "loss"), "train/loss", id="run-level scalar"),
        pytest.param(MetricKey(Stage.VAL, "f1", task="label"), "val/label/f1", id="task metric"),
        pytest.param(MetricKey(Stage.VAL, "f1/cat", task="label"), "val/label/f1/cat", id="per-class leaf"),
    ],
)
def test_writes_and_reads_back_the_same_key(key: MetricKey, text: str) -> None:
    assert str(key) == text
    assert MetricKey.parse(text) == key


def test_a_key_knows_its_family_and_leaf() -> None:
    key = MetricKey(Stage.VAL, "f1/cat", task="label")

    assert key.family == "val/label/f1"
    assert key.leaf == "cat"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"name": ""}, id="blank name"),
        pytest.param({"name": "f1//cat"}, id="empty segment"),
        pytest.param({"name": " f1"}, id="padded segment"),
        pytest.param({"name": "f1", "task": "a/b"}, id="separator in task"),
    ],
)
def test_refuses_a_malformed_key(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        MetricKey(Stage.VAL, **kwargs)


@pytest.mark.parametrize("text", ["", "loss", "epoch/1", "val", "val@hard/loss"], ids=repr)
def test_parse_refuses_text_outside_the_grammar(text: str) -> None:
    """A stage is a stage and nothing more: the head carrying anything else is not a key of ours."""
    with pytest.raises(ValueError):
        MetricKey.parse(text)


@pytest.mark.parametrize(
    ("key", "series"),
    [
        pytest.param(MetricKey(Stage.TRAIN, "loss"), "loss", id="the run's own number"),
        pytest.param(MetricKey(Stage.VAL, "f1", task="label"), "label/f1", id="a task's number"),
        pytest.param(MetricKey(Stage.TEST, "f1/cat", task="label"), "label/f1/cat", id="a leaf of one"),
    ],
)
def test_a_key_knows_what_every_stage_says_about_the_same_measurement(key: MetricKey, series: str) -> None:
    """The stage is the comparison, so what is left when it is dropped is the thing compared."""
    assert key.series == series


@pytest.mark.parametrize(
    ("logged", "headline"),
    [
        pytest.param("val/loss", "val/loss", id="a number is its own headline"),
        pytest.param("val/label/f1", "val/label/f1", id="so is a task's"),
        pytest.param("val/label/f1/mean", "val/label/f1", id="a family is read at its mean"),
        pytest.param("val/label/f1/cat", None, id="one class of it is detail"),
        pytest.param("epoch", None, id="not a measurement at all"),
        pytest.param("lr/backbone", None, id="nor is a knob of the optimizer"),
    ],
)
def test_the_reading_a_logged_key_contributes_to(logged: str, headline: str | None) -> None:
    """One rule for every at-a-glance view: a summary table and a progress table ask the same question."""
    found = MetricKey.headline(logged)

    assert (None if found is None else str(found)) == headline
