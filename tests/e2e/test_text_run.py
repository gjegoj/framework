"""A run whose input is not a picture: a column of sentences, read by the family that tokenized them.

Assembled through the composition root, because what the thread is about is three declarations meeting —
an encoder that turns one cell into several tensors, a collator that stacks each of them, and a backbone
called with them under the names its own family uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.build import build
from src.config import load_config
from src.experiment import run
from src.models import CompositeModel, LinearHead
from tests.support.declarations import smallest_run
from tests.support.table import write_table
from tests.support.text import WIDTH, text_family

if TYPE_CHECKING:
    from pathlib import Path

TASK, LENGTH = "species", 8


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def reading(table: Path, tmp_path: Path) -> dict[str, Any]:
    """The shipped skeleton with the picture taken out of it and the caption column put in its place."""
    family = str(text_family(tmp_path / "family"))
    declared = dict(smallest_run(table, tmp_path))
    declared["data"] = {**declared["data"], "inputs": {"text": {"column": "caption"}}}
    declared["preprocessing"] = {
        "name": "standard",
        "inputs": {"text": {"name": "text", "model_name": family, "max_length": LENGTH}},
    }
    # A run with no pixels declares no pipeline: the shipped chains interpolate an image size that a
    # declaration like this one does not carry.
    declared["transforms"] = {}
    declared["model"] = {"name": "composite", "backbone": {"name": "hf_text", "model_name": family}}
    return declared


def test_a_run_over_a_column_of_sentences_trains_and_reports_like_any_other(reading: dict[str, Any]) -> None:
    """Nothing downstream of the encoder knows text from pixels: the same kind, head, objective and report."""
    experiment = build(load_config(reading))

    run(experiment)

    logged = experiment.module.trainer.logged_metrics
    assert f"test/{TASK}/cross_entropy" in logged
    assert float(logged["test/loss"]) == pytest.approx(float(logged[f"test/{TASK}/cross_entropy"]))


def test_one_input_reaches_the_batch_as_the_several_tensors_its_family_reads(reading: dict[str, Any]) -> None:
    """What makes text different from a picture is here and nowhere else: one name, a tree of tensors."""
    data = build(load_config(reading)).data

    batch = next(iter(data.train_dataloader()))

    carried = batch.inputs["text"]
    assert isinstance(carried, dict)
    assert set(carried) == {"input_ids", "token_type_ids", "attention_mask"}
    assert all(tuple(one.shape) == (batch.count, LENGTH) for one in carried.values())


def test_the_head_is_sized_by_the_family_rather_than_by_anything_written_twice(reading: dict[str, Any]) -> None:
    """The width of a sentence encoder is the backbone's to publish, as a pooled picture's is."""
    model = build(load_config(reading)).module.learner.model

    assert isinstance(model, CompositeModel)
    head = model.heads[TASK]
    assert isinstance(head, LinearHead)
    assert head.projection.in_features == WIDTH
