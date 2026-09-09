"""Annotation: a task's step tensors become labels and a verdict, as its kind reads and draws them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from src import tasks as kinds
from src.core.entities import TaskFacts
from src.visualization import (
    Classification,
    Classifications,
    Image,
    Regression,
    SampleView,
    Segmentation,
    TaskView,
)
from src.visualization.annotators import ClassReading, DrawingKnobs, MulticlassReader
from tests.support.entities import a_task


def task_of(kind: kinds.TaskKind, names: list[str] | None = None) -> kinds.Task:
    return a_task(name="t", kind=kind, class_names=names)


def activated(kind: kinds.TaskKind, logits: torch.Tensor) -> torch.Tensor:
    """Push logits through the kind's own activation.

    A binary head's `[B, 1]` is squeezed to `[B]` before any consumer sees it, so a fixture shaped
    like logits would hide a reader that crashes on real output.
    """
    return kind.activation(TaskFacts())(logits)


def annotate(task: kinds.Task, logits: torch.Tensor, targets: torch.Tensor, **knobs: Any) -> SampleView:
    """Annotate one sample from what a head emits, activated the way the run activates it."""
    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    outputs = activated(task.kind, logits)
    task.kind.annotate(sample, task, outputs, targets, 0, DrawingKnobs(**knobs))
    return sample


def test_multiclass_argmaxes_and_judges() -> None:
    task = task_of(kinds.Classification(), ["cat", "dog"])

    sample = annotate(task, torch.tensor([[0.2, 2.0]]), torch.tensor([0]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "dog"
    assert pred.confidence == pytest.approx(0.858, abs=1e-3)  # softmax of the logits, not the logits
    assert sample.fields[("t", "gt")] == Classification(label="cat")
    assert sample.verdicts["t"].correct is False


def test_binary_thresholds_because_argmax_would_always_answer_class_zero() -> None:
    """A binary head emits one sigmoid value, so an argmax over it would always answer class zero."""
    task = task_of(kinds.BinaryClassification(), ["neg", "pos"])

    sample = annotate(task, torch.tensor([[2.0]]), torch.tensor([1]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "pos"
    assert pred.confidence == pytest.approx(0.881, abs=1e-3)
    assert sample.verdicts["t"].correct is True


def test_the_declared_threshold_reaches_the_reader_that_names_it() -> None:
    """A threshold no config could reach is no knob; the declared one is offered to every reader that names it."""
    task = task_of(kinds.BinaryClassification(), ["neg", "pos"])

    sample = annotate(task, torch.tensor([[2.0]]), torch.tensor([1]), threshold=0.95)

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "neg"


def test_multilabel_is_correct_only_when_the_whole_set_matches() -> None:
    task = task_of(kinds.MultilabelClassification(), ["a", "b", "c"])

    sample = annotate(task, torch.tensor([[2.0, -2.0, 1.5]]), torch.tensor([[1.0, 0.0, 0.0]]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classifications)
    assert [item.label for item in pred.classifications] == ["a", "c"]
    assert sample.verdicts["t"].correct is False  # 'a' matched, and one extra class is still a miss


def test_regression_scores_the_gap_and_returns_no_binary_verdict() -> None:
    task = task_of(kinds.Regression())

    sample = annotate(task, torch.tensor([[5.2]]), torch.tensor([4.0]))

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Regression)
    assert predicted.value == pytest.approx(5.2)
    assert sample.verdicts["t"].correct is None
    (score,) = sample.verdicts["t"].scores
    assert score.name == "mae"  # what metric_registry and the regression preset call it
    assert score.value == pytest.approx(1.2, abs=1e-5)


def test_segmentation_masks_every_present_class_and_skips_ignore_index() -> None:
    task = task_of(kinds.Segmentation(), ["bg", "cat"])
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 1, :2] = 4.0  # the top half is predicted 'cat'
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[0, :, :2] = 1  # the left half is 'cat'

    sample = annotate(task, logits, targets, ignore_index=0)

    gt = sample.fields[("t", "gt")]
    assert isinstance(gt, Segmentation)
    assert [entry.name for entry in gt.classes] == ["cat"]
    (score,) = sample.verdicts["t"].scores
    assert score.name == "iou"  # what metric_registry and the segmentation preset call it


def test_the_pages_iou_matches_the_metric_whose_name_it_borrows() -> None:
    """`ignore_index` has to drop the void *pixels*, not only the void class.

    A model cannot predict void, so every void pixel it labels lands in some real class's union and counts
    against it. Measured: a prediction correct on every valid pixel scored 0.70 on the page while
    `MulticlassJaccardIndex(ignore_index=...)` said 1.0 — two numbers under one name.
    """
    void = 3
    truth = torch.tensor([[[0, 1, 1, void], [0, 1, 1, void], [0, 0, void, void], [0, 0, void, void]]])
    perfect = truth.clone()
    perfect[truth == void] = 1  # correct everywhere it is allowed to be judged
    logits = torch.zeros(1, 4, 4, 4).scatter_(1, perfect.unsqueeze(1), 8.0)
    task = task_of(kinds.Segmentation(), ["bg", "cat", "dog", "void"])

    sample = annotate(task, logits, truth, ignore_index=void)

    (score,) = sample.verdicts["t"].scores
    reference = MulticlassJaccardIndex(num_classes=4, ignore_index=void, average="macro")
    assert score.value == pytest.approx(float(reference(perfect, truth)))
    assert score.value == pytest.approx(1.0)


def test_a_sample_with_nothing_left_to_judge_earns_no_score_rather_than_a_zero() -> None:
    """An empty union is not a score of zero — a zero would sort it as the worst on the page.

    It would also drag the slider's floor to 0, so the band a user narrows to find
    real mistakes would be calibrated on a sample that was never judged at all.
    """
    void = 2
    truth = torch.full((1, 4, 4), void, dtype=torch.long)
    logits = torch.zeros(1, 3, 4, 4)
    task = task_of(kinds.Segmentation(), ["bg", "cat", "void"])

    sample = annotate(task, logits, truth, ignore_index=void)

    assert sample.verdicts["t"].scores == ()


def test_maps_that_do_not_share_a_shape_are_refused_by_name() -> None:
    """numpy would broadcast `[1, W]` against `[H, W]` and report a perfect IoU.

    A head emitting at stride 8 against a full-resolution target is a real mistake,
    and the two ways it ended otherwise were a bare broadcast error naming neither
    side, or a silent 1.0 for a model that is not perfect.
    """
    task = task_of(kinds.Segmentation(), ["bg", "cat"])
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1] = 4.0

    with pytest.raises(ValueError, match=r"do not share a shape"):
        annotate(task, logits, torch.zeros(1, 4, 4, dtype=torch.long))


def test_an_unnamed_class_is_called_what_the_rest_of_the_run_calls_it() -> None:
    """The task's facts document the fallback and `loggers.report` uses it: `class{i}`.

    A bare index here would give one run two names for one class — `class3` on the
    metric leaves and the confusion matrix, `3` on the sample grid — so filtering a
    tracker by either finds half the story.
    """
    task = task_of(kinds.Classification())  # no class names declared

    sample = annotate(task, torch.tensor([[0.1, 0.2, 4.0]]), torch.tensor([2]))

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Classification)
    assert predicted.label == "class2"


def test_a_binary_dense_task_draws_a_foreground_mask() -> None:
    """The pair a registry keyed by (topology, objective) silently omitted."""
    task = task_of(kinds.BinarySegmentation(), ["background", "foreground"])
    logits = torch.full((1, 1, 4, 4), -4.0)
    logits[0, 0, :2] = 4.0
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[0, :2] = 1

    sample = annotate(task, logits, targets, ignore_index=0)

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Segmentation)
    assert [entry.name for entry in pred.classes] == ["foreground"]
    assert pred.classes[0].mask[:2].all()
    assert not pred.classes[0].mask[2:].any()


def test_a_global_binary_head_reads_after_its_channel_is_squeezed_away() -> None:
    """Regression: `sigmoid_probabilities` squeezes `[B, 1]` to `[B]`, so a sample arrives 0-d.

    Reading `scores[0]` there raised IndexError on every real binary run — the
    fixture that hid it was shaped like logits, not like an activated output.
    """
    task = task_of(kinds.BinaryClassification(), ["neg", "pos"])
    outputs = activated(kinds.BinaryClassification(), torch.tensor([[3.0], [-3.0]]))

    assert outputs.shape == (2,)  # the shape a real run hands the annotator

    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    task.kind.annotate(sample, task, outputs, torch.tensor([1, 0]), 0, DrawingKnobs())

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Classification)
    assert predicted.label == "pos"


def test_a_dense_binary_head_masks_the_whole_map_not_its_first_row() -> None:
    """Regression: the same squeeze turns `[B, 1, H, W]` into `[B, H, W]`.

    That case did not raise — `scores[0]` quietly took row 0 and produced an
    `(W,)` mask where an `(H, W)` one belongs, so the overlay was a stripe.
    """
    task = task_of(kinds.BinarySegmentation(), ["background", "foreground"])
    logits = torch.full((1, 1, 6, 8), -4.0)
    logits[0, 0, 3:] = 4.0
    outputs = activated(kinds.BinaryClassification(), logits)

    assert outputs.shape == (1, 6, 8)

    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    targets = torch.zeros(1, 6, 8, dtype=torch.long)
    task.kind.annotate(sample, task, outputs, targets, 0, DrawingKnobs(ignore_index=0))

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Segmentation)
    assert predicted.classes[0].mask.shape == (6, 8)
    assert predicted.classes[0].mask[3:].all()
    assert not predicted.classes[0].mask[:3].any()


def test_one_reader_serves_both_shapes_of_output() -> None:
    """The point of the split: multiclass argmaxes once, whatever shape it is given."""
    flat = MulticlassReader().read_output(np.array([0.2, 0.8]))
    spatial = MulticlassReader().read_output(np.array([[[0.2]], [[0.8]]]))

    assert isinstance(flat, ClassReading)
    assert isinstance(spatial, ClassReading)
    assert [entry.index for entry in flat.presences] == [1]
    assert [entry.index for entry in spatial.presences] == [1]
    assert flat.presences[0].where.shape == ()
    assert spatial.presences[0].where.shape == (1, 1)


def test_the_page_names_a_measure_the_way_the_framework_names_it() -> None:
    """One quantity, one name: the page said `miou` and `err` where the run logs `iou` and `mae`.

    Pinned against the registry itself, so renaming a metric there is caught here
    rather than by someone noticing two names for one number on two screens.
    """
    from src.metrics.registry import metric_registry
    from src.visualization.annotators import IOU, MAE

    assert IOU in metric_registry
    assert MAE in metric_registry


def test_a_mixed_pair_of_readings_is_refused_naming_both_sides() -> None:
    """One reader produces both readings, so a mixed pair is a wiring bug — named,
    not silently drawn. Uncovered until a mutation check showed nothing guarded it."""
    from src.visualization.annotators import ClassReading, GlobalDrawer, ValueReading

    truth = ClassReading(presences=(), singular=True)
    predicted = ValueReading(values=np.zeros(1))

    with pytest.raises(TypeError, match="ClassReading.*ValueReading.*one reader"):
        GlobalDrawer().annotate(SampleView(), TaskView("t"), truth, predicted)


def test_a_reading_kind_the_router_does_not_know_is_refused_by_name() -> None:
    """A new Reading member must not reach a run before ``annotate`` can route it.

    The union is the one place a kind exists; the ``match`` in ``annotate`` is the
    one place it is routed. This pins their agreement from the routing side, as
    the retired LABELLERS table test pinned it from the table side.
    """
    from dataclasses import dataclass

    from src.visualization.annotators import GlobalDrawer

    @dataclass
    class HeatReading:  # a stand-in for a future Reading member
        values: np.ndarray

    pair = HeatReading(values=np.zeros(1))
    with pytest.raises(TypeError, match="HeatReading"):
        GlobalDrawer().annotate(SampleView(), TaskView("t"), pair, pair)  # type: ignore[arg-type]


def test_a_pairing_a_drawer_cannot_draw_names_itself() -> None:
    """Unreachable while every kind pairs a reader with a drawer that draws it — and loud if one ever does not."""
    from src.visualization.annotators import DenseDrawer, ValueReading

    field = ValueReading(values=np.zeros((2, 2)))

    with pytest.raises(TypeError, match="DenseDrawer has no label for a ValueReading"):
        DenseDrawer().annotate(SampleView(), TaskView("t"), field, field)
