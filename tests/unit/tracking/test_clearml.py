"""ClearML behind Lightning's logger: keys become graphs, and a reading that draws is drawn.

The service is stubbed, so what is under test is the adapter's own decisions — which comparison a
key is turned into, what a matrix arrives with, and what happens when the far end is unreachable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.core import BYTES_PER_GIB, Bars, Matrix
from src.tracking import ClearMLTracker, DrawsBars, DrawsMatrix, KeepsFiles, KeepsRecord, RecordsSummary, ShowsPage
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

    @pytest.mark.parametrize(
        ("declared", "sent"),
        [
            pytest.param({}, {"pytorch": False}, id="nothing declared"),
            pytest.param(
                {"auto_connect_frameworks": {"matplotlib": False}},
                {"matplotlib": False, "pytorch": False},
                id="every other capture stays as declared",
            ),
            pytest.param({"auto_connect_frameworks": {"pytorch": False}}, {"pytorch": False}, id="already off"),
            pytest.param({"auto_connect_frameworks": False}, False, id="everything off stays so"),
            pytest.param({"auto_connect_frameworks": None}, None, id="null is everything off to ClearML"),
            pytest.param({"auto_connect_frameworks": {}}, {}, id="an empty mapping is everything off to ClearML"),
        ],
    )
    def test_clearml_is_not_left_to_capture_the_weights_itself(
        self, clearml: Recorded, declared: dict[str, Any], sent: object
    ) -> None:
        """Measured (spec 2026-09-25): left on, its PyTorch hook files every epoch's checkpoint as an output model
        and the one read back as an input model — a page with the current weights nowhere on it.

        A framework a mapping does not name it captures, so `pytorch` is added and the rest left alone — but a
        mapping that is empty it reads as nothing at all, and adding a key would turn the rest back on."""
        assert ClearMLTracker(**declared).experiment

        assert clearml.started["auto_connect_frameworks"] == sent

    @pytest.mark.parametrize(
        "declared",
        [
            pytest.param(True, id="everything on"),
            pytest.param({"pytorch": True}, id="pytorch on"),
            pytest.param({"pytorch": ["*.ckpt"]}, id="pytorch filtered by name"),
        ],
    )
    def test_a_declaration_asking_clearml_to_capture_the_weights_itself_is_refused(
        self, clearml: Recorded, declared: object
    ) -> None:
        """Which of a run's model files reach the service is `keep_suffixes`; a second answer would disagree."""
        with pytest.raises(ValueError, match=re.escape("Leave `pytorch` out of the mapping")):
            ClearMLTracker(auto_connect_frameworks=declared)

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


def model_file(directory: Path, name: str, size: int = 8) -> Path:
    """A file of a model, named as the export names it and as large as a test says — sparse, so a GiB is free."""
    path = directory / name
    with path.open("wb") as written:
        written.truncate(size)
    return path


def test_it_keeps_files_and_says_so_structurally(logger: ClearMLTracker) -> None:
    assert isinstance(logger, KeepsFiles)


@pytest.mark.parametrize(
    ("declared", "offered", "kept", "said"),
    [
        pytest.param(
            {},
            [("model.onnx", 8), ("model.onnx.data", 8)],
            ["model.onnx", "model.onnx.data"],
            (logging.INFO, "model.onnx.data is kept on the service."),
            id="every suffix, and a format whole",
        ),
        pytest.param(
            {"keep_suffixes": ["onnx"]},
            [("model.onnx", 8), ("model.onnx.data", 8)],
            ["model.onnx", "model.onnx.data"],
            (logging.INFO, "model.onnx.data is kept on the service."),
            id="what travels with a kept file goes with it",
        ),
        pytest.param(
            {"keep_suffixes": ["pt"]},
            [("epoch=3-step=16.ckpt", 8)],
            [],
            (logging.INFO, "epoch=3-step=16.ckpt stays on disk: `tracker.keep_suffixes` keeps pt."),
            id="a suffix not named",
        ),
        pytest.param(
            {"keep_suffixes": []},
            [("model.pt", 8)],
            [],
            (logging.INFO, "model.pt stays on disk: `tracker.keep_suffixes` keeps nothing."),
            id="an empty list keeps none",
        ),
        pytest.param(
            {"keep_max_gib": 12 / BYTES_PER_GIB},
            [("model.onnx", 8), ("model.onnx.data", 8)],
            [],
            (logging.WARNING, "it stays in {directory}."),
            id="the limit is on the whole format, not on each file",
        ),
        pytest.param(
            {"keep_max_gib": None},
            [("model.pt", BYTES_PER_GIB + 1)],
            ["model.pt"],
            (logging.INFO, "model.pt is kept on the service."),
            id="no limit",
        ),
        pytest.param(
            {},
            [("model.pt", BYTES_PER_GIB + 1)],
            [],
            (logging.WARNING, "model.pt is 1.00 GiB with what travels with it, above `tracker.keep_max_gib: 1.0`"),
            id="one GiB unless told otherwise",
        ),
    ],
)
def test_a_format_reaches_the_service_whole_or_not_at_all_as_the_run_declared(
    clearml: Recorded,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    declared: dict[str, Any],
    offered: list[tuple[str, int]],
    kept: list[str],
    said: tuple[int, str],
) -> None:
    """Every file that goes is waited for: this is the last step of a run, after `finalize`, and a line saying a
    file is on the service has to mean it is. One left on disk is named, with what decided it — at the level of
    what the run declared: a suffix left out is a choice, a format over the limit a warning."""
    path, *travelling = (model_file(tmp_path, name, size) for name, size in offered)
    level, line = said

    with caplog.at_level(logging.INFO):
        ClearMLTracker(**declared).log_file(path, travelling)

    logged = " ".join(one.getMessage() for one in caplog.records if one.levelno == level)
    assert list(clearml.artifacts) == kept and clearml.waited == kept
    assert line.format(directory=tmp_path) in logged


@pytest.mark.parametrize(
    ("declared", "refused_with"),
    [
        pytest.param({"keep_suffixes": [".pt"]}, "without their dot and in lower case", id="a suffix with its dot"),
        pytest.param({"keep_suffixes": ["ONNX"]}, "without their dot and in lower case", id="a suffix in another case"),
        pytest.param({"keep_suffixes": "pt"}, "without their dot and in lower case", id="a string, not a list"),
        pytest.param({"keep_max_gib": 0}, "so it is positive", id="a limit nothing fits under"),
    ],
)
def test_a_selection_it_cannot_follow_is_refused_where_the_run_is_built(
    clearml: Recorded, declared: dict[str, Any], refused_with: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(refused_with)):
        ClearMLTracker(**declared)


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        pytest.param("fails_to_upload", "the service is unreachable", id="raises"),
        pytest.param("declines_upload", "the service declined it", id="declines"),
    ],
)
@pytest.mark.parametrize(
    ("failing", "taken"),
    [
        pytest.param("model.onnx", [], id="the file a deployment opens"),
        pytest.param("model.onnx.data", ["model.onnx"], id="what travels with it"),
    ],
)
def test_the_first_file_the_service_does_not_take_is_named_and_stops_its_format(
    clearml: Recorded,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    failure: str,
    error: str,
    failing: str,
    taken: list[str],
) -> None:
    """Telemetry is not the run — the files are on disk whatever the service answers, as at `finalize` — and
    what travels with a file that did not arrive would be half a model there. The warning says how much of the
    format did arrive, and no line claims a file the service never took."""
    setattr(clearml, failure, failing)

    with caplog.at_level(logging.INFO):
        ClearMLTracker().log_file(model_file(tmp_path, "model.onnx"), [model_file(tmp_path, "model.onnx.data")])

    assert list(clearml.artifacts) == taken
    assert (
        f"ClearML could not keep {failing} ({error}); {len(taken)} of the 2 files of model.onnx reached it, and all "
        f"of them are still in {tmp_path}."
    ) in caplog.text
    assert f"{failing} is kept" not in caplog.text


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
