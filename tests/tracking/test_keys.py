"""One grammar for every reported value: ``stage[@split]/[task/]name``, read and written in one place."""

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
        pytest.param(MetricKey(Stage.VAL, "f1", split="val", task="label"), "val/label/f1", id="split equal to stage"),
        pytest.param(MetricKey(Stage.TEST, "map", split="2024", task="boxes"), "test@2024/boxes/map", id="own split"),
        pytest.param(MetricKey(Stage.TRAIN, "lr", split="hard"), "train@hard/lr", id="split without task"),
    ],
)
def test_writes_and_reads_back_the_same_key(key: MetricKey, text: str) -> None:
    assert str(key) == text
    assert (
        MetricKey.parse(text) == key
        if key.split != key.stage.value
        else MetricKey.parse(text) == MetricKey(key.stage, key.name, task=key.task)
    )


def test_a_key_knows_its_family_and_leaf() -> None:
    key = MetricKey(Stage.VAL, "f1/cat", task="label")

    assert key.family == "val/label/f1"
    assert key.leaf == "cat"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"name": ""}, id="blank name"),
        pytest.param({"name": "f1//cat"}, id="empty segment"),
        pytest.param({"name": "f1@cat"}, id="separator in name"),
        pytest.param({"name": "f1", "task": "a/b"}, id="separator in task"),
        pytest.param({"name": "f1", "split": "a/b"}, id="separator in split"),
        pytest.param({"name": "f1", "split": ""}, id="blank split"),
    ],
)
def test_refuses_a_malformed_key(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        MetricKey(Stage.VAL, **kwargs)


@pytest.mark.parametrize("text", ["", "loss", "epoch/1", "val"], ids=repr)
def test_parse_refuses_text_outside_the_grammar(text: str) -> None:
    with pytest.raises(ValueError):
        MetricKey.parse(text)
