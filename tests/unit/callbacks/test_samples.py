"""A batch of samples as a page in the tracker: when one is drawn, and what it holds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import pytest

from src.callbacks.samples import SampleGrid
from src.core import DatasetInfo, InputInfo, TargetInfo
from src.data import DataModule
from src.experiment import Experiment
from src.tasks.regression import Regression as RegressionTask
from src.tasks.segmentation import DenseOutput
from src.training import TrainingData
from tests.support.pages import PageRecorder
from tests.unit.callbacks.conftest import prepared

SHOWING = {"_target_": "tests.support.pages.PageRecorder"}
NUMBERS = {"_target_": "tests.support.pages.NumbersOnly"}


def drawing(**declared: Any) -> list[dict[str, Any]]:
    return [{"name": "samples", "every_n_epochs": 1, **declared}]


def fitted(declaration: Mapping[str, Any], **overrides: Any) -> PageRecorder:
    """A run that trains and is tested, and the backend that kept every page it drew."""
    built = prepared(declaration, tracker=SHOWING, **overrides)
    built.trainer.fit(built.module, datamodule=built.data)
    built.trainer.test(built.module, datamodule=built.data)
    return _recorder(built)


def titles(shown: PageRecorder) -> list[str]:
    return [title for title, _, _ in shown.pages]


def _recorder(built: Experiment) -> PageRecorder:
    return next(one for one in built.trainer.loggers if isinstance(one, PageRecorder))


class TestWhenAPageIsDrawn:
    def test_one_page_per_stage_the_run_declared(self, declaration: Mapping[str, Any]) -> None:
        shown = fitted(declaration, callbacks=drawing(stages=["val", "test"]))

        assert sorted(titles(shown)) == ["samples/test", "samples/val"]

    def test_a_stage_left_out_draws_nothing(self, declaration: Mapping[str, Any]) -> None:
        assert titles(fitted(declaration, callbacks=drawing(stages=["test"]))) == ["samples/test"]

    def test_only_the_batch_the_run_named_is_drawn(self, declaration: Mapping[str, Any]) -> None:
        """A fixed batch rather than a lucky one: two pages of a run compare only if it is the same samples."""
        assert fitted(declaration, callbacks=drawing(stages=["train"], batch_index=99)).pages == []

    def test_the_cadence_is_counted_in_epochs(self, declaration: Mapping[str, Any]) -> None:
        shown = fitted(declaration, epochs=3, callbacks=drawing(every_n_epochs=2, stages=["val"]))

        assert [iteration for _, _, iteration in shown.pages] == [0, 2]

    def test_the_test_stage_draws_whatever_the_cadence_says(self, declaration: Mapping[str, Any]) -> None:
        """It runs once, and Lightning reports its epoch as the fit's final count — so a cadence would make
        whether a test page exists at all depend on how many epochs the run happened to have."""
        assert "samples/test" in titles(fitted(declaration, epochs=1, callbacks=drawing(every_n_epochs=5)))

    def test_the_sanity_check_draws_nothing(self, declaration: Mapping[str, Any]) -> None:
        """An untrained network has no answer to where the model is wrong, and the page would land under
        the same title and iteration as the first real epoch's."""
        built = prepared(
            declaration,
            tracker=SHOWING,
            callbacks=drawing(stages=["val"]),
            trainer={"accelerator": "cpu", "num_sanity_val_steps": 1, "enable_progress_bar": False},
        )
        built.trainer.fit(built.module, datamodule=built.data)

        assert [iteration for _, _, iteration in _recorder(built).pages] == [0]


class TestWhatThePageHolds:
    def test_a_cell_per_sample_it_was_asked_for(self, declaration: Mapping[str, Any]) -> None:
        shown = fitted(declaration, callbacks=drawing(stages=["test"], num_images=2))

        assert shown.pages[0][1].count('class="cell') == 2

    def test_it_draws_what_the_run_learned_rather_than_only_reporting_a_number(
        self, declaration: Mapping[str, Any]
    ) -> None:
        page = fitted(declaration, callbacks=drawing(stages=["test"])).pages[0][1]

        assert "cat" in page or "dog" in page
        assert "data-verdicts=" in page

    def test_a_batch_smaller_than_the_page_is_not_padded(self, declaration: Mapping[str, Any]) -> None:
        page = fitted(declaration, callbacks=drawing(stages=["test"], num_images=99)).pages[0][1]

        assert 0 < page.count('class="cell') <= 2


class TestWhenThereIsNowhereToShowIt:
    def test_a_backend_that_shows_no_pages_is_told_once_rather_than_once_a_stage(
        self, declaration: Mapping[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Lightning runs setup once for the fit and again for the test, so the plain thing says it twice."""
        built = prepared(declaration, epochs=2, tracker=NUMBERS, callbacks=drawing())
        with caplog.at_level("WARNING"):
            built.trainer.fit(built.module, datamodule=built.data)
            built.trainer.test(built.module, datamodule=built.data)

        assert sum("can carry a page" in record.message for record in caplog.records) == 1


class TestDeclaration:
    def test_a_stage_this_framework_does_not_have_is_refused_with_the_ones_it_does(self) -> None:
        """The name a run reaches for first: every library spells this one differently, and here it is `val`."""
        with pytest.raises(ValueError, match="validation"):
            SampleGrid(stages=["validation"])

    @pytest.mark.parametrize(("knob", "value"), [("num_images", 0), ("every_n_epochs", 0), ("batch_index", -1)])
    def test_a_knob_that_could_only_draw_nothing_is_refused(self, knob: str, value: int) -> None:
        with pytest.raises(ValueError, match=knob):
            SampleGrid(**{knob: value})  # type: ignore[arg-type]


class TestWhatItCannotDraw:
    """Said once, and never at the cost of the run — the promise the class docstring makes."""

    def test_a_pipeline_this_framework_did_not_prepare_is_named(
        self, declaration: Mapping[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing then knows how a picture was normalised, so nothing can show it as the file held it."""
        built = prepared(declaration)

        with caplog.at_level("WARNING"):
            SampleGrid().setup(L.Trainer(logger=False), built.module, "fit")

        assert any("pipeline this framework prepares" in record.message for record in caplog.records)

    def test_an_input_that_declares_no_statistics_is_named(
        self, declaration: Mapping[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The silent version of this is a run that simply never draws, with nothing said about why."""
        built = prepared(declaration)
        trainer = L.Trainer(logger=False)
        trainer.datamodule = TrainingData(_Blind())  # type: ignore[attr-defined]

        with caplog.at_level("WARNING"):
            SampleGrid().setup(trainer, built.module, "fit")

        assert any("normalised" in record.message for record in caplog.records)

    def test_a_task_nothing_draws_is_named(
        self, declaration: Mapping[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A number at every pixel is a heat map, which the display vocabulary has no label for yet.

        That the rest of the run still draws is held where the decision is made, in the gallery's own
        tests; what is held here is that a run is told which task it lost and why.
        """
        built = prepared(declaration)
        built.trainer.datamodule = built.data  # type: ignore[attr-defined]
        depth = type("DepthMap", (DenseOutput, RegressionTask), {})("depth", TargetInfo())
        built.module.learner.tasks = {**built.module.learner.tasks, "depth": depth}

        with caplog.at_level("WARNING"):
            SampleGrid().setup(built.trainer, built.module, "fit")

        assert any("number at every pixel" in record.message for record in caplog.records)

    def test_a_page_that_cannot_be_built_costs_the_page_and_not_the_run(
        self, declaration: Mapping[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The loop calls the grid inside its own step, so anything escaping ends the fit at whatever
        epoch first went wrong — a full disk, an unreachable service, a mask at the wrong resolution."""
        built = prepared(declaration, tracker=SHOWING, callbacks=drawing(stages=["train"]))
        recorder = _recorder(built)
        recorder.fails = True

        with caplog.at_level("WARNING"):
            built.trainer.fit(built.module, datamodule=built.data)

        assert built.trainer.state.finished
        assert recorder.pages == []
        assert sum("stopped drawing" in record.message for record in caplog.records) == 1

    def test_it_stops_drawing_into_a_run_that_is_over(self, declaration: Mapping[str, Any]) -> None:
        """The module keeps the watcher, so a grid still holding its first trainer would write a second
        run's pages into the first run's tracker, under the first run's epoch number."""
        built = prepared(declaration, tracker=SHOWING, callbacks=drawing(stages=["test"]))
        built.trainer.fit(built.module, datamodule=built.data)
        built.trainer.test(built.module, datamodule=built.data)
        drawn = len(_recorder(built).pages)

        second = prepared(declaration, tracker=SHOWING, callbacks=[])
        second.trainer.test(built.module, datamodule=built.data)

        assert len(_recorder(built).pages) == drawn
        assert _recorder(second).pages == []


class TestTheCadence:
    def test_it_counts_the_epochs_this_stage_runs_rather_than_the_trainer_s(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """A run declaring `check_val_every_n_epoch: 2` validates only on odd epochs, so a cadence
        tested for divisibility against the epoch number lands on none of them — measured, that pairing
        drew no validation page at all for any even cadence."""
        built = prepared(
            declaration,
            epochs=6,
            tracker=SHOWING,
            callbacks=drawing(every_n_epochs=2, stages=["val"]),
            trainer={"accelerator": "cpu", "check_val_every_n_epoch": 2, "enable_progress_bar": False},
        )
        built.trainer.fit(built.module, datamodule=built.data)

        assert [iteration for _, _, iteration in _recorder(built).pages] == [1, 5]


class _Blind(DataModule):
    """A prepared pipeline with one input, which says nothing about how it was normalised."""

    @property
    def preprocessor(self) -> Any:
        raise NotImplementedError

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(inputs={"image": InputInfo(shape=None)}, targets={})

    def setup(self, splits: Any) -> None:
        raise NotImplementedError

    def dataset(self, split: str) -> Any:
        raise NotImplementedError
