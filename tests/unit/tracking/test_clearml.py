"""ClearML behind Lightning's logger: keys become graphs, and a reading that draws is drawn.

The service is stubbed, so what is under test is the adapter's own decisions — which comparison a
key is turned into, what a matrix arrives with, and what happens when the far end is unreachable.
"""

from __future__ import annotations

import logging

import pytest
import torch

from src.core import Matrix
from src.tracking import ClearMLTracker, DrawsMatrix
from tests.unit.tracking.conftest import Recorded


@pytest.fixture
def logger(clearml: Recorded) -> ClearMLTracker:
    return ClearMLTracker(project_name="pets", task_name="a-run")


@pytest.mark.parametrize(
    ("key", "drawn"),
    [
        pytest.param("train/loss", ("loss", "train"), id="the stages of one number share a graph"),
        pytest.param("val/species/f1", ("species/f1", "val"), id="a task's number, the same way"),
        pytest.param("val/species/f1/cat", ("val/species/f1", "cat"), id="a family compares its own leaves"),
        pytest.param(
            "train/contribution/species/dice",
            ("train/contribution/species", "dice"),
            id="what the objective is made of, per term",
        ),
        pytest.param("lr/backbone", ("lr", "backbone"), id="outside the grammar: the leaves are the comparison"),
        pytest.param("epoch", ("epoch", "value"), id="a number that is nothing else"),
    ],
)
def test_a_key_becomes_the_graph_and_the_line_it_belongs_on(
    logger: ClearMLTracker, clearml: Recorded, key: str, drawn: tuple[str, str]
) -> None:
    logger.log_metrics({key: 1.0}, step=2)

    assert clearml.scalars == [(*drawn, 1.0, 2)]


def test_a_reading_with_no_step_of_its_own_is_the_first_iteration(logger: ClearMLTracker, clearml: Recorded) -> None:
    logger.log_metrics({"train/loss": 0.5})

    assert clearml.scalars == [("loss", "train", 0.5, 0)]


def test_it_draws_matrices_and_says_so_structurally(logger: ClearMLTracker) -> None:
    """A backend qualifies for a shape by having the method; the router asks nothing else."""
    assert isinstance(logger, DrawsMatrix)


def test_a_matrix_arrives_with_its_axes_and_the_names_of_its_rows(logger: ClearMLTracker, clearml: Recorded) -> None:
    matrix = Matrix(torch.eye(2) / 3, xaxis="Predicted", yaxis="True", labels=("cat", "dog"))

    logger.log_matrix("val/species/confusion_matrix", matrix, 1)

    drawn = clearml.matrices[0]
    assert drawn["title"] == "val/species/confusion_matrix" and drawn["iteration"] == 1
    assert drawn["xlabels"] == drawn["ylabels"] == ["cat", "dog"]
    assert (drawn["xaxis"], drawn["yaxis"]) == ("Predicted", "True")
    assert drawn["matrix"][0][0] == pytest.approx(0.333), "cells are read, not computed with"


def test_a_matrix_of_a_task_that_named_nothing_is_drawn_without_labels(
    logger: ClearMLTracker, clearml: Recorded
) -> None:
    logger.log_matrix("val/species/confusion_matrix", Matrix(torch.eye(2), xaxis="P", yaxis="T"), 0)

    assert clearml.matrices[0]["xlabels"] is None


class TestTheRun:
    def test_every_knob_of_the_service_forwards_verbatim(self, clearml: Recorded) -> None:
        """Its own documentation stays the reference: nothing is relayed by a parameter of ours."""
        ClearMLTracker(project_name="pets", output_uri="s3://bucket")

        assert clearml.started["project_name"] == "pets" and clearml.started["output_uri"] == "s3://bucket"

    def test_a_tag_that_says_nothing_is_not_a_tag(self, clearml: Recorded) -> None:
        """Tags are written as interpolations, and a group that is off leaves an empty string behind."""
        ClearMLTracker(tags=["adamw", "", "adamw", "lr=0.001"])

        assert clearml.started["tags"] == ["adamw", "lr=0.001"]

    def test_a_fresh_run_rather_than_whatever_the_service_would_reuse(
        self, logger: ClearMLTracker, clearml: Recorded
    ) -> None:
        assert clearml.started["reuse_last_task_id"] is False

    def test_the_run_answers_with_the_identity_the_service_gave_it(self, logger: ClearMLTracker) -> None:
        assert (logger.name, logger.version) == ("a-run", "abc123")

    def test_the_whole_configuration_reaches_the_run_as_its_hyperparameters(
        self, logger: ClearMLTracker, clearml: Recorded
    ) -> None:
        logger.log_hyperparams({"lr": 1e-3})

        assert clearml.connected == {"lr": 1e-3}

    def test_what_was_recorded_is_pushed_when_the_run_ends(self, logger: ClearMLTracker, clearml: Recorded) -> None:
        logger.finalize("success")

        assert clearml.flushes == 1

    def test_an_unreachable_service_at_the_end_does_not_take_the_results_with_it(
        self, logger: ClearMLTracker, clearml: Recorded, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Telemetry is not the run: a fit that finished must not fail on its way to saying so."""
        clearml.fails_to_flush = True

        with caplog.at_level(logging.WARNING):
            logger.finalize("success")

        assert "unreachable" in caplog.text
