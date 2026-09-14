"""ClearML behind Lightning's logger: keys become graphs, and a reading that draws is drawn.

The service is stubbed, so what is under test is the adapter's own decisions — which comparison a
key is turned into, what a matrix arrives with, and what happens when the far end is unreachable.
"""

from __future__ import annotations

import logging

import pytest
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.core import Bars, Matrix
from src.tracking import ClearMLTracker, DrawsBars, DrawsMatrix, KeepsRecord, RecordsSummary, ShowsPage
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


def test_it_keeps_a_summary_table_and_says_so_structurally(logger: ClearMLTracker) -> None:
    assert isinstance(logger, RecordsSummary)


def test_a_headline_number_goes_where_the_service_keeps_those(logger: ClearMLTracker, clearml: Recorded) -> None:
    """Off the iteration axis on purpose: one value for the whole run is not a line of one point."""
    logger.record_summary("label/f1", 0.75)

    assert clearml.singles == {"label/f1": 0.75}


def test_a_matrix_of_a_task_that_named_nothing_is_drawn_without_labels(
    logger: ClearMLTracker, clearml: Recorded
) -> None:
    logger.log_matrix("val/species/confusion_matrix", Matrix(torch.eye(2), xaxis="P", yaxis="T"), 0)

    assert clearml.matrices[0]["xlabels"] is None


class TestTheRun:
    def test_every_knob_of_the_service_forwards_verbatim(self, clearml: Recorded) -> None:
        """Its own documentation stays the reference: nothing is relayed by a parameter of ours."""
        assert ClearMLTracker(project_name="pets", output_uri="s3://bucket").experiment

        assert clearml.started["project_name"] == "pets" and clearml.started["output_uri"] == "s3://bucket"

    def test_a_tag_that_says_nothing_is_not_a_tag(self, clearml: Recorded) -> None:
        """Tags are written as interpolations, and a group that is off leaves an empty string behind."""
        assert ClearMLTracker(tags=["adamw", "", "adamw", "lr=0.001"]).experiment

        assert clearml.started["tags"] == ["adamw", "lr=0.001"]

    def test_a_fresh_run_rather_than_whatever_the_service_would_reuse(
        self, logger: ClearMLTracker, clearml: Recorded
    ) -> None:
        assert logger.experiment and clearml.started["reuse_last_task_id"] is False

    def test_the_run_on_the_service_is_created_by_the_first_thing_reported_to_it(
        self, logger: ClearMLTracker, clearml: Recorded
    ) -> None:
        """A constructor runs on every device; an experiment must not be a side effect of one."""
        assert clearml.started == {}

        logger.log_metrics({"train/loss": 1.0})

        assert clearml.started["project_name"] == "pets"

    def test_a_device_that_only_follows_creates_no_run_of_its_own(
        self, clearml: Recorded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four devices used to mean four experiments there, three of them empty."""
        monkeypatch.setattr(rank_zero_only, "rank", 1)
        follower = ClearMLTracker(project_name="pets", task_name="a-run")

        follower.log_metrics({"train/loss": 1.0})
        follower.log_hyperparams({"lr": 1e-3})
        follower.finalize("success")

        assert clearml.started == {} and follower.version == ""

    def test_the_run_answers_with_the_identity_the_service_gave_it(self, logger: ClearMLTracker) -> None:
        assert (logger.name, logger.version) == ("a-run", "abc123")

    def test_the_whole_configuration_reaches_the_run_as_its_hyperparameters(
        self, logger: ClearMLTracker, clearml: Recorded
    ) -> None:
        logger.log_hyperparams({"lr": 1e-3})

        assert clearml.connected == {"lr": 1e-3}

    def test_what_was_recorded_is_pushed_when_the_run_ends(self, logger: ClearMLTracker, clearml: Recorded) -> None:
        logger.log_metrics({"train/loss": 1.0})

        logger.finalize("success")

        assert clearml.flushes == 1

    def test_an_unreachable_service_at_the_end_does_not_take_the_results_with_it(
        self, logger: ClearMLTracker, clearml: Recorded, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Telemetry is not the run: a fit that finished must not fail on its way to saying so."""
        logger.log_metrics({"train/loss": 1.0})
        clearml.fails_to_flush = True

        with caplog.at_level(logging.WARNING):
            logger.finalize("success")

        assert "unreachable" in caplog.text


def test_a_page_is_shipped_as_media_so_the_service_renders_it_in_place(
    logger: ClearMLTracker, clearml: Recorded
) -> None:
    """The one detail that decides it: an html extension opens the page inside the panel, where any
    other filing gives a file to download, and nobody downloads a file to look at a batch."""
    logger.log_html("samples/val", "<html>a page</html>", iteration=3)

    (shipped,) = clearml.media
    assert shipped["file_extension"] == "html"
    assert (shipped["title"], shipped["iteration"]) == ("samples/val", 3)
    assert shipped["read"] == "<html>a page</html>"


def test_it_shows_pages_and_says_so_structurally(logger: ClearMLTracker) -> None:
    assert isinstance(logger, ShowsPage)


def test_a_balance_is_one_grouped_chart_with_a_series_per_split(logger: ClearMLTracker, clearml: Recorded) -> None:
    """Grouped rather than stacked: the question is how the splits compare on one class, and stacking
    puts that comparison inside a single column."""
    bars = Bars(
        series=("train", "val"),
        values=((3.0, 1.0), (1.0, 1.0)),
        labels=("cat", "dog"),
        xaxis="class",
        yaxis="count",
    )

    logger.log_bars("dataset/species", bars, iteration=0)

    assert [one["series"] for one in clearml.histograms] == ["train", "val"]
    assert {one["mode"] for one in clearml.histograms} == {"group"}
    assert clearml.histograms[0]["values"] == [3.0, 1.0]
    assert clearml.histograms[0]["xlabels"] == ["cat", "dog"]


def test_it_draws_bars_and_says_so_structurally(logger: ClearMLTracker) -> None:
    assert isinstance(logger, DrawsBars)


def test_the_record_of_what_a_run_produced_is_kept_as_something_fetched_back_whole(
    logger: ClearMLTracker, clearml: Recorded
) -> None:
    """An export record is neither a number nor a picture: whoever deploys the model reads it from a
    script, so the service has to hand it back as one object rather than render it."""
    logger.log_record("model", {"inputs": [{"name": "image"}]})

    assert clearml.artifacts == {"model": {"inputs": [{"name": "image"}]}}


def test_it_keeps_records_and_says_so_structurally(logger: ClearMLTracker) -> None:
    assert isinstance(logger, KeepsRecord)


def test_a_run_that_names_where_its_pages_go_sends_them_there_rather_than_to_the_file_server(
    clearml: Recorded,
) -> None:
    """A page is the one reading here that becomes a file, and the service uploads it to one place only.

    Read in clearml 2.1.10: a debug sample goes to `api.files_server` and nothing in `Task.init`
    moves it — `output_uri` is documented for models and artifacts, and no key of `clearml.conf`
    reaches it. Only the logger's own destination does, so this is the one knob of ours that is not
    forwarded, and the service must never be handed it.
    """
    ClearMLTracker(media_uri="s3://bucket/media").log_html("samples/val", "<p>a page</p>", 0)

    assert clearml.destinations == ["s3://bucket/media"]
    assert "media_uri" not in clearml.started


def test_a_run_that_names_nowhere_leaves_the_service_the_destination_it_chose(clearml: Recorded) -> None:
    """Declaring nothing is the ordinary case, and it has to stay the service's own decision.

    What is read is that the service was never told, rather than what it was told: handed `None` it
    would record a destination of `None`, which reads exactly like never having been asked.
    """
    ClearMLTracker().log_html("samples/val", "<p>a page</p>", 0)

    assert clearml.destinations == []
