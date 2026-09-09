"""``MetricsProgressBar``: a live table of current and best values under the bar."""

from __future__ import annotations

import pytest

from src.callbacks.progress import MetricHistory, MetricsProgressBar, row_key
from src.training.ports import DeclaresMetricDirections
from tests.support.lightning import quiet_trainer


@pytest.mark.parametrize(
    ("logged", "row"),
    [
        ("train/loss", "train/loss"),
        ("val/label/f1", "val/label/f1"),
        ("val/label/iou/mean", "val/label/iou"),
        ("val/label/iou/cat", None),
        ("epoch", None),
        ("val/label/iou/very/deep", None),
    ],
)
def test_a_logged_key_normalizes_to_its_table_row_or_drops(logged: str, row: str | None) -> None:
    """Scalars and vector means feed the table; per-class leaves and stage-less keys are noise."""
    assert row_key(logged) == row


def test_observing_twice_records_the_step_delta() -> None:
    history = MetricHistory({"val/label/f1": True})

    history.observe("val/label/f1", 0.5)
    history.observe("val/label/f1", 0.7)

    assert history.current["val/label/f1"] == pytest.approx(0.7)
    assert history.step_deltas["val/label/f1"] == pytest.approx(0.2)


def test_best_follows_the_declared_direction() -> None:
    """higher_is_better=True: a lower value must not overwrite the best."""
    history = MetricHistory({"val/label/f1": True})

    history.observe("val/label/f1", 0.5)
    history.observe("val/label/f1", 0.7)
    history.observe("val/label/f1", 0.6)

    assert history.best["val/label/f1"] == pytest.approx(0.7)
    assert history.best_deltas["val/label/f1"] == pytest.approx(0.2)


def test_what_the_module_does_not_declare_is_read_as_a_loss() -> None:
    """Metrics are declared and losses are not, so lower is knowledge here rather than a guess.

    It holds for a loss part and for a term a decorated model adds at runtime
    alike — neither has to be announced for its best to be tracked.
    """
    history = MetricHistory({"val/label/f1": True})

    assert history.direction("train/loss") == "min"
    assert history.direction("train/label/ce") == "min"
    assert history.direction("train/label/kl") == "min"

    history.observe("train/label/ce", 1.0)
    history.observe("train/label/ce", 0.4)

    assert history.best["train/label/ce"] == pytest.approx(0.4)


def test_a_metric_declared_directionless_still_tracks_no_best() -> None:
    """The half of the rule doing real work: a confusion matrix has no better side to show."""
    history = MetricHistory({"val/label/confusion_matrix": None})

    history.observe("val/label/confusion_matrix", 0.5)
    history.observe("val/label/confusion_matrix", 0.9)

    assert "val/label/confusion_matrix" not in history.best


@pytest.fixture(scope="module")
def after_a_run() -> MetricsProgressBar:
    """The bar of a finished fit-and-test, so both claims read one run rather than two."""
    from functools import partial

    import torch

    from src.training import TrainingData, TrainingModule
    from tests.support.entities import a_task
    from tests.support.fakes import a_composite
    from tests.support.tables import in_memory_pipeline

    task = a_task()
    module = TrainingModule(
        model=a_composite(2),
        tasks=[task],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )
    data_module, _ = in_memory_pipeline()
    data = TrainingData(data_module, batch_size=2)
    bar = MetricsProgressBar()
    trainer = quiet_trainer(callbacks=[bar], enable_progress_bar=True)

    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data)
    return bar


def test_a_run_with_the_bar_fits_and_tests_without_a_tty(after_a_run: MetricsProgressBar) -> None:
    """Rich disables itself off-tty; the bar must survive a full fit+test regardless."""
    assert after_a_run is not None


def test_the_table_sees_every_stage_the_run_reported(after_a_run: MetricsProgressBar) -> None:
    """The table has a column per stage, and only `train/loss` is logged with `prog_bar=True`.

    Fed from the bar's own `get_metrics` it would show that one value and leave
    Val and Test permanently blank — measured, empty at both.
    """
    seen = set(after_a_run.history.current)

    assert {"train/loss", "val/loss", "test/loss"} <= seen


def test_the_finished_table_shows_train_and_val_beside_test(after_a_run: MetricsProgressBar) -> None:
    """A Test column means nothing alone — it is there to be read against what training reached.

    Lightning empties `callback_metrics` between `fit` and `test`, so a table built
    from the keys of one refresh blanks Train and Val exactly when Test arrives.
    """
    table = after_a_run.table()
    series = [str(cell) for cell in table.columns[0].cells]
    row = series.index("loss")
    values = {column.header: str(list(column.cells)[row]) for column in table.columns[1:]}

    assert values["Train"], f"the Train cell is blank: {values}"
    assert values["Val"], f"the Val cell is blank: {values}"
    assert values["Test"], f"the Test cell is blank: {values}"


def test_a_loss_part_gets_a_best_like_the_total_it_belongs_to(after_a_run: MetricsProgressBar) -> None:
    """The optimizer descends the weighted sum of exactly these parts, so lower is knowledge, not a guess.

    Left without one, a part would sit beside a total that has a best and show
    none — which reads as "this one has no direction" rather than "nobody said".
    """
    assert after_a_run.history.direction("train/label/ce") == "min"
    assert "train/label/ce" in after_a_run.history.best


def test_nothing_but_a_loss_arrives_undeclared(after_a_run: MetricsProgressBar) -> None:
    """The premise behind reading undeclared as `min`, pinned to the keys that rely on it.

    A fourth kind of scalar reaching the table would be signed as a loss in
    silence. Naming them here makes that a red test instead.
    """
    module = after_a_run.trainer.lightning_module
    assert isinstance(module, DeclaresMetricDirections)
    undeclared = set(after_a_run.history.current) - set(module.metric_directions())

    assert undeclared == {
        "train/loss",
        "val/loss",
        "test/loss",
        "train/label/ce",
        "val/label/ce",
        "test/label/ce",
    }


def test_a_stage_less_observation_never_becomes_a_row_of_the_table() -> None:
    """A key with no stage has no series to sit under: ``epoch`` is dropped on the Lightning path by
    ``row_key``, and the table must drop it on the public path too — hand-splitting the key at the
    separator had turned it into a row with an empty name."""
    bar = MetricsProgressBar()
    bar.history.observe("epoch", 3.0)
    bar.history.observe("val/label/ce", 0.5)

    series = [str(cell) for cell in bar.table().columns[0].cells]

    assert series == ["label/ce"]
