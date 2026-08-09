"""``ClearMLLogger``: one backend task, keys split by the grammar's owner."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.core import Curve, Matrix
from src.core.entities import Bars, Spread, ValueDistribution
from src.core.ports import CurveLogger, MatrixLogger

# The class itself needs no backend — `clearml` is imported inside `__init__`.
from src.loggers import ClearMLLogger


@pytest.fixture
def clearml_stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    recorded = SimpleNamespace(
        scalars=[],
        matrices=[],
        curves=[],
        single_values=[],
        media=[],
        histograms=[],
        figures=[],
        flushed=0,
        init_kwargs=None,
        connected=None,
    )

    class _Backend:
        def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
            recorded.scalars.append((title, series, value, iteration))

        def report_single_value(self, name: str, value: float) -> None:
            recorded.single_values.append((name, value))

        def report_confusion_matrix(self, **kwargs: Any) -> None:
            recorded.matrices.append(kwargs)

        def report_scatter2d(self, **kwargs: Any) -> None:
            recorded.curves.append(kwargs)

        def report_media(self, **kwargs: Any) -> None:
            recorded.media.append(kwargs)

        def report_histogram(self, **kwargs: Any) -> None:
            recorded.histograms.append(kwargs)

        def report_plotly(self, **kwargs: Any) -> None:
            recorded.figures.append(kwargs)

    class _Task:
        name = "run"
        id = "abc123"

        @classmethod
        def init(cls, **kwargs: Any) -> _Task:
            recorded.init_kwargs = kwargs
            return cls()

        def get_logger(self) -> _Backend:
            return _Backend()

        def connect(self, mapping: dict[str, Any]) -> None:
            recorded.connected = mapping

        def flush(self) -> None:
            recorded.flushed += 1

    # A real spec, because a stub without one is not merely incomplete: `accelerate`
    # probes for clearml with `importlib.util.find_spec`, which raises rather than
    # answering "no" when a module in `sys.modules` has none. Whether it raised
    # depended on whether something else had imported accelerate first.
    stub = SimpleNamespace(Task=_Task, __spec__=ModuleSpec("clearml", loader=None))
    monkeypatch.setitem(sys.modules, "clearml", stub)
    return recorded


def build_logger(**kwargs: Any) -> Any:
    from src.loggers import ClearMLLogger

    return ClearMLLogger(**kwargs)


def test_scalars_split_stage_first_so_stages_share_a_graph(clearml_stub: SimpleNamespace) -> None:
    """val/label/f1 and train/label/f1 must land on one plot as two series — losses too."""
    logger = build_logger()

    logger.log_metrics({"val/label/f1": 0.5, "train/loss": 1.0}, step=2)

    assert ("label/f1", "val", 0.5, 2) in clearml_stub.scalars
    assert ("loss", "train", 1.0, 2) in clearml_stub.scalars


def test_the_adapter_satisfies_both_artifact_ports_structurally(clearml_stub: SimpleNamespace) -> None:
    logger = build_logger()

    assert isinstance(logger, MatrixLogger)
    assert isinstance(logger, CurveLogger)


def test_a_matrix_reaches_the_backend_with_labels_and_axes(clearml_stub: SimpleNamespace) -> None:
    logger = build_logger()
    matrix = Matrix(value=torch.eye(2), xaxis="Predicted", yaxis="True", labels=("cat", "dog"))

    logger.log_matrix(title="val/label/cm", matrix=matrix, iteration=1)

    reported = clearml_stub.matrices[0]
    assert reported["title"] == "val/label/cm"
    assert reported["xlabels"] == ["cat", "dog"]
    assert (reported["xaxis"], reported["yaxis"]) == ("Predicted", "True")


def test_a_curve_draws_every_series_of_one_figure(clearml_stub: SimpleNamespace) -> None:
    curve = Curve(
        x=(torch.tensor([0.0]), torch.tensor([0.1])),
        y=(torch.tensor([1.0]), torch.tensor([0.9])),
        xaxis="Recall",
        yaxis="Precision",
        series=("cat", "dog"),
    )

    build_logger().log_curve(title="val/label/pr", curve=curve, iteration=2)

    assert [entry["series"] for entry in clearml_stub.curves] == ["cat", "dog"]
    assert clearml_stub.curves[0]["xaxis"] == "Recall"


def test_upstream_task_knobs_forward_verbatim(clearml_stub: SimpleNamespace) -> None:
    """Every ``Task.init`` option stays reachable without adapter changes."""
    build_logger(project_name="pets", output_uri="s3://bucket")

    assert clearml_stub.init_kwargs["project_name"] == "pets"
    assert clearml_stub.init_kwargs["output_uri"] == "s3://bucket"


def test_a_class_balance_is_one_grouped_series_per_stage(clearml_stub: SimpleNamespace) -> None:
    """Grouped, not stacked: the question is how the splits compare on one class.

    Stacking would put that comparison inside a single column.
    """
    bars = Bars(
        series=("train", "val"),
        values=((30.0, 10.0), (6.0, 4.0)),
        labels=("cat", "dog"),
        xaxis="class",
        yaxis="count",
    )

    build_logger().log_bars(title="dataset/label", bars=bars, iteration=0)

    assert [one["series"] for one in clearml_stub.histograms] == ["train", "val"]
    assert clearml_stub.histograms[0]["values"] == [30.0, 10.0]
    assert clearml_stub.histograms[0]["xlabels"] == ["cat", "dog"]
    assert clearml_stub.histograms[0]["mode"] == "group"


def test_a_box_is_built_from_the_summary_and_never_from_the_values(
    clearml_stub: SimpleNamespace,
) -> None:
    """ClearML draws no box, so this is the one artifact the framework draws itself.

    plotly accepts a box described entirely by its quartiles, so no column has to be
    carried into memory for it — which is what makes counting every stage affordable.
    The fences are the observed extremes; `Spread` says so rather than implying Tukey.
    """
    measured = ValueDistribution(
        count=4, mean=2.5, deviation=1.3, minimum=1.0, q25=1.75, median=2.5, q75=3.25, maximum=4.0
    )
    spread = Spread(series=("train",), boxes=(measured,), xaxis="stage", yaxis="value")

    build_logger().log_spread(title="dataset/age", spread=spread, iteration=0)

    (reported,) = clearml_stub.figures
    (box,) = reported["figure"].data
    # The series is also the box's position on the axis: without one every stage is
    # drawn at zero, so they stack on top of each other instead of standing apart.
    assert box.x == ("train",)
    assert (box.name, box.q1, box.median, box.q3) == ("train", (1.75,), (2.5,), (3.25,))
    assert (box.lowerfence, box.upperfence, box.mean) == ((1.0,), (4.0,), (2.5,))
    # `y` is where a box's raw samples would go, and it is empty: only the summary
    # crossed the port. `x` carries the stage's name, which is a position, not data.
    assert box.y is None


REPORTING = sorted(name for name in vars(ClearMLLogger) if name.startswith("log_") or name == "finalize")
"""Every method that talks to the backend, read off the class so a new one is covered."""


@pytest.mark.parametrize("name", REPORTING)
def test_every_reporting_method_is_guarded_at_the_logger(name: str) -> None:
    """The guard belongs here, not with the callers, and this is what keeps it that way.

    `report_metric` runs on every rank at epoch end, so a port left unguarded
    uploads one copy of itself per device — which is exactly what `log_curve` did
    until it was found. Read off the class rather than listed, so the next artifact
    port cannot be added without one.
    """
    guarded = getattr(ClearMLLogger, name)

    # `functools.wraps` copies the name and the module off the wrapped function, so
    # neither can tell a guarded method from a bare one. It does not copy `__globals__`,
    # which still belong to whichever module defined the wrapper.
    home = guarded.__globals__["__name__"]
    assert home.endswith("rank_zero"), f"{name} is missing @rank_zero_only (wrapper from {home})"


def test_a_guarded_call_sends_nothing_off_rank_zero(
    clearml_stub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the marker means what the test above assumes: on rank 1 the backend hears nothing.

    Structure alone would pass for any `functools.wraps` decorator; this pins the
    behaviour the structure stands for.
    """
    logger = build_logger()
    bars = Bars(series=("train",), values=((1.0,),), labels=("cat",), xaxis="class", yaxis="count")
    monkeypatch.setattr(rank_zero_only, "rank", 1)

    logger.log_bars(title="dataset/label", bars=bars, iteration=0)
    logger.log_single_value(name="test/label/f1", value=1.0)

    assert clearml_stub.histograms == []
    assert clearml_stub.single_values == []


def test_each_stage_gets_its_own_place_on_the_axis(clearml_stub: SimpleNamespace) -> None:
    """Boxes are compared by standing side by side; stacked, they hide one another.

    A `go.Box` built from precomputed quartiles has no position unless it is given
    one, and plotly draws every positionless trace at zero — which rendered three
    stages as one shape and labelled the axis with a bare "0".
    """
    measured = ValueDistribution(
        count=4, mean=2.5, deviation=1.3, minimum=1.0, q25=1.75, median=2.5, q75=3.25, maximum=4.0
    )
    spread = Spread(
        series=("train", "val", "test"),
        boxes=(measured, measured, measured),
        xaxis="stage",
        yaxis="value",
    )

    build_logger().log_spread(title="dataset/age", spread=spread, iteration=0)

    (reported,) = clearml_stub.figures
    assert [box.x for box in reported["figure"].data] == [("train",), ("val",), ("test",)]


def test_finalize_flushes_and_swallows_backend_failures(clearml_stub: SimpleNamespace) -> None:
    """A telemetry hiccup at teardown must never take the run's results with it."""
    logger = build_logger()

    logger.finalize("success")
    assert clearml_stub.flushed == 1

    logger._task.flush = _raise
    logger.finalize("success")  # must not raise


def _raise() -> None:
    raise ConnectionError("backend gone")


def test_hyperparams_connect_to_the_task(clearml_stub: SimpleNamespace) -> None:
    logger = build_logger()

    logger.log_hyperparams({"lr": 0.1})

    assert clearml_stub.connected == {"lr": 0.1}


def test_reuse_last_task_id_is_a_forwardable_knob(clearml_stub: SimpleNamespace) -> None:
    """The docstring promises every Task.init knob forwards; this one must not be pinned."""
    build_logger(reuse_last_task_id=True)

    assert clearml_stub.init_kwargs["reuse_last_task_id"] is True


def test_an_uncompleted_curve_is_refused_by_name(clearml_stub: SimpleNamespace) -> None:
    """A direct port caller skipping the router must hear about series, not a cryptic zip error."""
    curve = Curve(x=(torch.tensor([0.0]),), y=(torch.tensor([1.0]),), xaxis="x", yaxis="y")

    with pytest.raises(ValueError, match="series"):
        build_logger().log_curve(title="val/label/pr", curve=curve, iteration=0)


def test_single_values_round_for_the_summary_table(clearml_stub: SimpleNamespace) -> None:
    build_logger().log_single_value("label/f1", 0.333333)

    assert clearml_stub.single_values == [("label/f1", 0.333)]


def test_the_architecture_joins_the_declared_tags(clearml_stub: SimpleNamespace) -> None:
    """Config cannot supply this one: the key naming an architecture differs per backbone family.

    So assembly asks the model and offers the answer here, where it becomes the tag
    a run is found by whatever built it.
    """
    build_logger(tags=["timm", "adamw"], architecture="unet-resnet34")

    assert clearml_stub.init_kwargs["tags"] == ["timm", "adamw", "unet-resnet34"]


def test_a_tag_that_resolved_to_nothing_is_dropped(clearml_stub: SimpleNamespace) -> None:
    """Tags are written as interpolations, and a group that is off leaves an empty string behind."""
    build_logger(tags=["timm", "", "lr=0.001", ""], architecture=None)

    assert clearml_stub.init_kwargs["tags"] == ["timm", "lr=0.001"]


def test_a_tag_declared_twice_appears_once(clearml_stub: SimpleNamespace) -> None:
    """A timm run names its family and its architecture the same on a bare backbone."""
    build_logger(tags=["resnet18", "adamw"], architecture="resnet18")

    assert clearml_stub.init_kwargs["tags"] == ["resnet18", "adamw"]


def test_a_page_ships_as_media_because_that_is_what_clearml_embeds(
    clearml_stub: SimpleNamespace,
) -> None:
    """An artifact upload would give a file to download; nobody downloads a file to look at a batch."""
    logger = build_logger()

    logger.log_html(title="samples/val", html="<html>x</html>", iteration=3)

    entry = clearml_stub.media[0]
    assert entry["title"] == "samples/val"
    assert entry["iteration"] == 3
    assert entry["file_extension"] == "html"
    assert entry["stream"].read() == "<html>x</html>"
