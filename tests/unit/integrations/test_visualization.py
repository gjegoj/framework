"""The seam: a batch, a step and a run's tasks become the values a page is drawn from."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torchmetrics import JaccardIndex

from src.core import Batch, InputInfo, Normalization, StepOutput, TargetInfo, TensorShape, require_tensor
from src.core.taxonomy import Axis, Modality
from src.integrations.visualization import AnyAnnotator, Gallery, annotator_for, drawn_input, vocabulary_of
from src.tasks.registry import task_registry
from src.tasks.regression import Regression as RegressionTask
from src.tasks.segmentation import DenseOutput
from src.visualization import Classification, Classifications, Image, Regression, SampleView, Segmentation
from tests.support.tasks import CLASSES, COUNT, specimen

GREY = Normalization(mean=(0.5,), std=(0.25,))
COLOUR = Normalization(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))

PICTURE = "photo"


def normalized(pixels: np.ndarray, normalization: Normalization) -> torch.Tensor:
    """What the pixel pipeline hands the model: float, channels first, statistics removed."""
    scaled = torch.as_tensor(pixels, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(normalization.mean).view(1, -1, 1, 1)
    std = torch.tensor(normalization.std).view(1, -1, 1, 1)
    return (scaled - mean) / std


def image_info(normalization: Normalization = COLOUR) -> InputInfo:
    channels = len(normalization.mean)
    shape = TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(channels, 4, 5))
    return InputInfo(shape=shape, modality=Modality.IMAGE, normalization=normalization)


class TestDrawnInput:
    def test_the_picture_a_page_draws_is_the_one_that_says_how_to_undo_its_statistics(self) -> None:
        assert drawn_input({"tabular": InputInfo(shape=None), PICTURE: image_info()}) == (PICTURE, COLOUR)

    def test_a_run_with_nothing_to_draw_says_so_rather_than_guessing(self) -> None:
        assert drawn_input({"tabular": InputInfo(shape=None)}) is None

    def test_the_first_declared_picture_wins_so_the_choice_does_not_move_between_runs(self) -> None:
        found = drawn_input({"a": image_info(), "b": image_info()})
        assert found is not None
        assert found[0] == "a"


class TestAnnotators:
    @pytest.mark.parametrize("kind", sorted(task_registry))
    def test_every_kind_a_run_can_declare_says_something_about_a_sample(self, kind: str) -> None:
        """A specimen of each kind, annotated: both sides of the cell filled, and a verdict recorded."""
        task, output, batch = specimen(kind)
        view = blank()

        drawing(task).annotate(view, task, predictions(task, output), task.metric_view(batch), index=0)

        assert (task.name, "gt") in view.fields
        assert (task.name, "pred") in view.fields
        assert task.name in view.verdicts

    def test_a_task_deciding_a_number_at_every_pixel_is_left_off_rather_than_drawn_wrong(self) -> None:
        """No shipped kind pairs them, but a kind of one's own may: a heat map is not a set of labels."""
        depth = type("DepthMap", (DenseOutput, RegressionTask), {})

        assert annotator_for(depth("depth", TargetInfo())) is None

    def test_one_class_per_sample_reads_as_the_class_the_model_scored_highest(self) -> None:
        task, _, _ = specimen("classification")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([[0.1, 0.7, 0.2]]), torch.tensor([2]), index=0)

        said = view.fields[("t", "pred")]
        assert isinstance(said, Classification)
        assert (said.label, said.confidence) == ("dog", pytest.approx(0.7))
        assert view.fields[("t", "gt")] == Classification("bird")

    def test_a_class_no_vocabulary_names_is_called_what_the_metric_leaves_call_it(self) -> None:
        """A binary task declares no words, and its page must not invent any of its own."""
        task, _, _ = specimen("binary_classification")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([0.8]), torch.tensor([1]), index=0)

        said = view.fields[("t", "pred")]
        assert isinstance(said, Classification)
        assert (said.label, said.confidence) == ("class1", pytest.approx(0.8))

    def test_any_number_of_labels_reads_as_every_one_above_the_line(self) -> None:
        task, _, _ = specimen("multilabel_classification")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([[0.9, 0.1, 0.6]]), torch.tensor([[1, 0, 0]]), index=0)

        predicted = view.fields[("t", "pred")]
        assert isinstance(predicted, Classifications)
        assert [one.label for one in predicted.classifications] == ["cat", "bird"]
        assert view.verdicts["t"].correct is False

    def test_a_sample_both_sides_agree_on_is_correct(self) -> None:
        task, _, _ = specimen("classification")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([[0.1, 0.7, 0.2]]), torch.tensor([1]), index=0)

        assert view.verdicts["t"].correct is True

    def test_a_decision_per_pixel_reads_as_one_mask_per_class_and_an_overlap(self) -> None:
        task, output, batch = specimen("segmentation")
        view = blank()

        drawing(task).annotate(view, task, predictions(task, output), task.metric_view(batch), index=0)

        drawn = view.fields[("t", "gt")]
        assert isinstance(drawn, Segmentation)
        assert {one.name for one in drawn.classes} <= set(CLASSES.values())
        assert [score.name for score in view.verdicts["t"].scores] == ["iou"]

    def test_a_prediction_that_matches_every_pixel_earns_the_whole_overlap(self) -> None:
        task, _, _ = specimen("segmentation")
        truth = torch.zeros(1, 4, 5, dtype=torch.long)
        truth[0, :, 3:] = 1
        scores = torch.nn.functional.one_hot(truth, num_classes=3).permute(0, 3, 1, 2).float()
        view = blank()

        drawing(task).annotate(view, task, scores, truth, index=0)

        assert view.verdicts["t"].scores[0].value == pytest.approx(1.0)

    def test_a_class_neither_side_shows_says_nothing_about_the_overlap(self) -> None:
        """A class absent from both sides is not a zero: averaging one in would sort a sample the model
        got entirely right below one it got half right."""
        task, _, _ = specimen("segmentation")
        truth = torch.zeros(1, 4, 5, dtype=torch.long)
        scores = torch.nn.functional.one_hot(truth, num_classes=3).permute(0, 3, 1, 2).float()
        view = blank()

        drawing(task).annotate(view, task, scores, truth, index=0)

        assert view.verdicts["t"].scores[0].value == pytest.approx(1.0), "one class shown, and it matched"

    @pytest.mark.parametrize(
        ("truth", "said"),
        [
            pytest.param([[1, 1, 0, 0]], [[1, 0, 0, 0]], id="half of what was claimed"),
            pytest.param([[1, 1, 1, 1]], [[0, 0, 0, 0]], id="a model that predicted nothing"),
            pytest.param([[1, 0, 1, 0]], [[1, 1, 1, 1]], id="a model that predicted everything"),
        ],
    )
    def test_the_page_measures_a_binary_mask_the_way_the_run_measures_it(
        self, truth: list[list[int]], said: list[list[int]]
    ) -> None:
        """The negative side is drawn and not scored. Averaging the background in is a different
        quantity: a model predicting nothing on a sparse mask scored 0.48 that way against the 0.0 the
        run reports, so the samples the page exists to find were the ones it hid best.
        """
        task, _, _ = specimen("binary_segmentation")
        wanted, predicted = torch.tensor(truth).float(), torch.tensor(said).float()
        view = blank()

        drawing(task).annotate(view, task, predicted, wanted, index=0)

        metric = JaccardIndex(task="binary")
        metric.update(predicted, wanted.long())
        assert view.verdicts["t"].scores[0].value == pytest.approx(float(metric.compute()))

    def test_a_sample_neither_side_claims_anything_on_earns_no_score_at_all(self) -> None:
        """A zero would sort it as the worst thing on the page; it is simply a sample with nothing to say."""
        task, _, _ = specimen("binary_segmentation")
        empty = torch.zeros(1, 4, 5)
        view = blank()

        drawing(task).annotate(view, task, empty, empty, index=0)

        assert view.verdicts["t"].scores == ()

    def test_the_background_of_a_binary_mask_is_not_painted_over_the_picture(self) -> None:
        """It is the complement of the shape, so drawing it covers everything the page is about."""
        task, _, _ = specimen("binary_segmentation")
        truth = torch.tensor([[1, 1, 0, 0]]).float()
        view = blank()

        drawing(task).annotate(view, task, truth, truth, index=0)

        drawn = view.fields[("t", "gt")]
        assert isinstance(drawn, Segmentation)
        assert [one.name for one in drawn.classes] == ["class1"]

    def test_a_class_that_holds_nowhere_is_not_drawn_as_an_empty_mask(self) -> None:
        task, _, _ = specimen("segmentation")
        truth = torch.zeros(1, 4, 5, dtype=torch.long)
        scores = torch.nn.functional.one_hot(truth, num_classes=3).permute(0, 3, 1, 2).float()
        view = blank()

        drawing(task).annotate(view, task, scores, truth, index=0)

        drawn = view.fields[("t", "gt")]
        assert isinstance(drawn, Segmentation)
        assert [one.name for one in drawn.classes] == ["cat"]

    def test_a_chip_carries_the_score_the_model_gave_that_class(self) -> None:
        """Only the side that expressed one: the truth carries no confidence, because it expressed none."""
        task, _, _ = specimen("classification")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([[0.1, 0.65, 0.25]]), torch.tensor([0]), index=0)

        said, truth = view.fields[("t", "pred")], view.fields[("t", "gt")]
        assert isinstance(said, Classification) and isinstance(truth, Classification)
        assert said.confidence == pytest.approx(0.65)
        assert truth.confidence is None

    def test_a_number_is_drawn_as_itself_and_scored_by_how_far_it_missed(self) -> None:
        task, _, _ = specimen("regression")
        view = blank()

        drawing(task).annotate(view, task, torch.tensor([2.0]), torch.tensor([3.5]), index=0)

        assert view.fields[("t", "pred")] == Regression(2.0)
        assert view.verdicts["t"].scores[0].name == "mae"
        assert view.verdicts["t"].scores[0].value == pytest.approx(1.5)


class TestVocabulary:
    def test_a_declared_vocabulary_travels_in_the_order_it_was_declared(self) -> None:
        task, _, _ = specimen("classification")

        assert vocabulary_of(task) == ("cat", "dog", "bird")

    def test_a_binary_task_declares_none_and_still_has_two_names(self) -> None:
        """Without them a page showing only one of the two sides would recolour it, which is the drift
        a fixed batch exists to make visible rather than to cause."""
        task, _, _ = specimen("binary_classification")

        assert vocabulary_of(task) == ("class0", "class1")

    def test_a_number_has_no_classes_to_name(self) -> None:
        task, _, _ = specimen("regression")

        assert vocabulary_of(task) == ()


class TestGallery:
    def test_the_picture_comes_back_in_the_colours_the_run_took_it_from(self) -> None:
        """Every byte of it, not merely a picture that looks about right.

        Exactly, because the nearest byte is a choice and the obvious alternative is wrong: cutting
        toward zero rather than rounding lands a level low on most values, and comparing a page
        against the original file is the whole reason the statistics are undone at all.
        """
        pixels = np.random.default_rng(0).integers(0, 256, size=(COUNT, 4, 5, 3), dtype=np.uint8)

        views = gallery().views(batch_of(normalized(pixels, COLOUR)), step(), count=COUNT)

        assert np.array_equal(views[0].picture.pixels, pixels[0])

    def test_one_grey_plane_is_shown_as_grey_rather_than_refused(self) -> None:
        grey = np.random.default_rng(1).integers(0, 256, size=(COUNT, 4, 5, 1), dtype=np.uint8)

        drawn = gallery(GREY).views(batch_of(normalized(grey, GREY)), step(), count=1)[0]

        assert drawn.picture.pixels.shape == (4, 5, 3)
        assert np.array_equal(drawn.picture.pixels[..., 0], drawn.picture.pixels[..., 2])

    def test_a_cell_names_the_file_it_came_from(self) -> None:
        views = gallery().views(batch_of(torch.zeros(COUNT, 3, 4, 5)), step(), count=1)

        assert views[0].picture.source == "/data/0.png"

    def test_a_pipeline_that_carried_no_paths_still_draws(self) -> None:
        plain = Batch(inputs={PICTURE: torch.zeros(COUNT, 3, 4, 5)}, count=COUNT)

        assert gallery().views(plain, step(), count=1)[0].picture.source is None

    def test_it_draws_no_more_than_it_was_asked_for(self) -> None:
        assert len(gallery().views(batch_of(torch.zeros(COUNT, 3, 4, 5)), step(), count=2)) == 2

    def test_a_task_the_step_did_not_answer_for_is_left_off_and_the_cell_is_still_drawn(self) -> None:
        views = gallery().views(batch_of(torch.zeros(COUNT, 3, 4, 5)), StepOutput(loss=None), count=1)

        assert views[0].fields == {}
        assert views[0].picture is not None

    def test_the_whole_vocabulary_travels_with_the_page_so_colours_hold_across_it(self) -> None:
        assert gallery().classes == {"t": tuple(CLASSES[index] for index in sorted(CLASSES))}

    def test_a_task_nothing_draws_is_named_rather_than_silently_missing(self) -> None:
        depth = type("DepthMap", (DenseOutput, RegressionTask), {})("depth", TargetInfo())
        task, _, _ = specimen("classification")

        built = Gallery.of({task.name: task, "depth": depth}, PICTURE, COLOUR)

        assert built.undrawable == ("depth",)
        assert set(built.annotators) == {"t"}


def drawing(task: Any) -> AnyAnnotator:
    """The annotator for a task, refused here so every test below reads as one statement."""
    found = annotator_for(task)
    assert found is not None, f"{type(task).__name__} has nothing that draws it"
    return found


def predictions(task: Any, output: Any) -> torch.Tensor:
    return require_tensor(task.postprocess(output), name=task.name)


def blank() -> SampleView:
    return SampleView(picture=Image(pixels=np.zeros((2, 2, 3), dtype=np.uint8)))


def gallery(normalization: Normalization = COLOUR) -> Gallery:
    task, _, _ = specimen("classification")
    return Gallery.of({task.name: task}, PICTURE, normalization)


def batch_of(pixels: torch.Tensor) -> Batch:
    return Batch(
        inputs={PICTURE: pixels},
        count=COUNT,
        metadata={"cells": [{PICTURE: f"/data/{index}.png"} for index in range(COUNT)]},
    )


def step() -> StepOutput:
    return StepOutput(
        loss=None,
        predictions={"t": torch.tensor([[0.1, 0.7, 0.2]] * COUNT)},
        targets={"t": torch.tensor([1] * COUNT)},
    )
