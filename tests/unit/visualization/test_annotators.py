"""Annotation: a task's step tensors become labels and a verdict, per axis pair."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from src.core import InputTopology, Objective, OutputTopology, Task
from src.core.entities import TargetFacts
from src.tasks.registry import objective_registry
from src.visualization import (
    Classification,
    Classifications,
    Image,
    Regression,
    SampleView,
    Segmentation,
)
from src.visualization.annotators import (
    ClassReading,
    MulticlassAnnotation,
    build_annotators,
)
from tests.support.entities import a_task


def task_of(topology: OutputTopology, objective: Objective, names: list[str] | None = None) -> Task:
    return a_task(name="t", output_topology=topology, objective=objective, class_names=names)


def activated(objective: Objective, logits: torch.Tensor) -> torch.Tensor:
    """Push logits through the framework's own activation for that objective.

    Tests used to hand-write the shape an annotator receives, and the two drifted:
    a binary head's `[B, 1]` is squeezed to `[B]` before any consumer sees it, so
    a fixture shaped like logits hid a reader that crashed on real output.
    """
    return objective_registry.create(objective).build_activation(TargetFacts())(logits)


def annotate(task: Task, logits: torch.Tensor, targets: torch.Tensor, **knobs: object) -> SampleView:
    """Annotate one sample from what a head emits, activated the way the run activates it."""
    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    outputs = activated(task.objective, logits)
    build_annotators([task], **knobs)["t"].annotate(sample, task, outputs, targets, index=0)
    return sample


def test_multiclass_argmaxes_and_judges() -> None:
    task = task_of(OutputTopology.GLOBAL, Objective.MULTICLASS, ["cat", "dog"])

    sample = annotate(task, torch.tensor([[0.2, 2.0]]), torch.tensor([0]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "dog"
    assert pred.confidence == pytest.approx(0.858, abs=1e-3)  # softmax of the logits, not the logits
    assert sample.fields[("t", "gt")] == Classification(label="cat")
    assert sample.verdicts["t"].correct is False


def test_binary_thresholds_because_argmax_would_always_answer_class_zero() -> None:
    """A binary head emits one sigmoid value; the reference caught this and so do we."""
    task = task_of(OutputTopology.GLOBAL, Objective.BINARY, ["neg", "pos"])

    sample = annotate(task, torch.tensor([[2.0]]), torch.tensor([1]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "pos"
    assert pred.confidence == pytest.approx(0.881, abs=1e-3)
    assert sample.verdicts["t"].correct is True


def test_the_declared_threshold_reaches_the_reader_that_names_it() -> None:
    """The reference declared thresholds no config could reach; here one is offered to all."""
    task = task_of(OutputTopology.GLOBAL, Objective.BINARY, ["neg", "pos"])

    sample = annotate(task, torch.tensor([[2.0]]), torch.tensor([1]), threshold=0.95)

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classification)
    assert pred.label == "neg"


def test_multilabel_is_correct_only_when_the_whole_set_matches() -> None:
    task = task_of(OutputTopology.GLOBAL, Objective.MULTILABEL, ["a", "b", "c"])

    sample = annotate(task, torch.tensor([[2.0, -2.0, 1.5]]), torch.tensor([[1.0, 0.0, 0.0]]))

    pred = sample.fields[("t", "pred")]
    assert isinstance(pred, Classifications)
    assert [item.label for item in pred.classifications] == ["a", "c"]
    assert sample.verdicts["t"].correct is False  # 'a' matched, and one extra class is still a miss


def test_regression_scores_the_gap_and_returns_no_binary_verdict() -> None:
    task = task_of(OutputTopology.GLOBAL, Objective.CONTINUOUS)

    sample = annotate(task, torch.tensor([[5.2]]), torch.tensor([4.0]))

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Regression)
    assert predicted.value == pytest.approx(5.2)
    assert sample.verdicts["t"].correct is None
    (score,) = sample.verdicts["t"].scores
    assert score.name == "mae"  # what metric_registry and the regression preset call it
    assert score.value == pytest.approx(1.2, abs=1e-5)


def test_segmentation_masks_every_present_class_and_skips_ignore_index() -> None:
    task = task_of(OutputTopology.DENSE, Objective.MULTICLASS, ["bg", "cat"])
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

    A model cannot predict void, so every void pixel it labels lands in some real
    class's union and counts against it. Measured before the fix: a prediction
    correct on every valid pixel scored 0.70 on the page while
    `MulticlassJaccardIndex(ignore_index=...)` said 1.0 — so the slider a user drags
    to find the worst samples was calibrated on a different number from the one the
    epoch report shows under the same name.
    """
    void = 3
    truth = torch.tensor([[[0, 1, 1, void], [0, 1, 1, void], [0, 0, void, void], [0, 0, void, void]]])
    perfect = truth.clone()
    perfect[truth == void] = 1  # correct everywhere it is allowed to be judged
    logits = torch.zeros(1, 4, 4, 4).scatter_(1, perfect.unsqueeze(1), 8.0)
    task = task_of(OutputTopology.DENSE, Objective.MULTICLASS, ["bg", "cat", "dog", "void"])

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
    task = task_of(OutputTopology.DENSE, Objective.MULTICLASS, ["bg", "cat", "void"])

    sample = annotate(task, logits, truth, ignore_index=void)

    assert sample.verdicts["t"].scores == ()


def test_maps_that_do_not_share_a_shape_are_refused_by_name() -> None:
    """numpy would broadcast `[1, W]` against `[H, W]` and report a perfect IoU.

    A head emitting at stride 8 against a full-resolution target is a real mistake,
    and the two ways it ended otherwise were a bare broadcast error naming neither
    side, or a silent 1.0 for a model that is not perfect.
    """
    task = task_of(OutputTopology.DENSE, Objective.MULTICLASS, ["bg", "cat"])
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1] = 4.0

    with pytest.raises(ValueError, match=r"do not share a shape"):
        annotate(task, logits, torch.zeros(1, 4, 4, dtype=torch.long))


def test_an_unnamed_class_is_called_what_the_rest_of_the_run_calls_it() -> None:
    """`Task.class_names` documents the fallback and `core.reporting` uses it: `class{i}`.

    A bare index here would give one run two names for one class — `class3` on the
    metric leaves and the confusion matrix, `3` on the sample grid — so filtering a
    tracker by either finds half the story.
    """
    task = task_of(OutputTopology.GLOBAL, Objective.MULTICLASS)  # no class names declared

    sample = annotate(task, torch.tensor([[0.1, 0.2, 4.0]]), torch.tensor([2]))

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Classification)
    assert predicted.label == "class2"


def test_a_binary_dense_task_draws_a_foreground_mask() -> None:
    """The pair a registry keyed by (topology, objective) silently omitted."""
    task = task_of(OutputTopology.DENSE, Objective.BINARY, ["background", "foreground"])
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
    task = task_of(OutputTopology.GLOBAL, Objective.BINARY, ["neg", "pos"])
    outputs = activated(Objective.BINARY, torch.tensor([[3.0], [-3.0]]))

    assert outputs.shape == (2,)  # the shape a real run hands the annotator

    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    build_annotators([task])["t"].annotate(sample, task, outputs, torch.tensor([1, 0]), index=0)

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Classification)
    assert predicted.label == "pos"


def test_a_dense_binary_head_masks_the_whole_map_not_its_first_row() -> None:
    """Regression: the same squeeze turns `[B, 1, H, W]` into `[B, H, W]`.

    That case did not raise — `scores[0]` quietly took row 0 and produced an
    `(W,)` mask where an `(H, W)` one belongs, so the overlay was a stripe.
    """
    task = task_of(OutputTopology.DENSE, Objective.BINARY, ["background", "foreground"])
    logits = torch.full((1, 1, 6, 8), -4.0)
    logits[0, 0, 3:] = 4.0
    outputs = activated(Objective.BINARY, logits)

    assert outputs.shape == (1, 6, 8)

    sample = SampleView(media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))})
    targets = torch.zeros(1, 6, 8, dtype=torch.long)
    build_annotators([task], ignore_index=0)["t"].annotate(sample, task, outputs, targets, index=0)

    predicted = sample.fields[("t", "pred")]
    assert isinstance(predicted, Segmentation)
    assert predicted.classes[0].mask.shape == (6, 8)
    assert predicted.classes[0].mask[3:].all()
    assert not predicted.classes[0].mask[:3].any()


def test_one_reader_serves_both_topologies_of_an_objective() -> None:
    """The point of the split: multiclass argmaxes once, whatever shape it is given."""
    flat = MulticlassAnnotation().read_output(np.array([0.2, 0.8]))
    spatial = MulticlassAnnotation().read_output(np.array([[[0.2]], [[0.8]]]))

    assert isinstance(flat, ClassReading)
    assert isinstance(spatial, ClassReading)
    assert [entry.index for entry in flat.presences] == [1]
    assert [entry.index for entry in spatial.presences] == [1]
    assert flat.presences[0].where.shape == ()
    assert spatial.presences[0].where.shape == (1, 1)


def test_metric_learning_is_skipped_with_its_reason_not_a_shrug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence here was the reference's worst case; 'no annotator' and 'nothing to show' differ."""
    task = task_of(OutputTopology.GLOBAL, Objective.MULTICLASS, ["a"])
    embedded = a_task(name="emb", objective=Objective.METRIC)

    with caplog.at_level(logging.INFO):
        built = build_annotators([task, embedded])

    assert "t" in built
    assert "emb" not in built
    assert any("emb" in record.message and "metric" in record.message for record in caplog.records)


def test_a_dense_regression_says_what_it_cannot_draw_yet(caplog: pytest.LogCaptureFixture) -> None:
    """Depth and heatmaps need a Label the IR has not got; half-drawing them would lie."""
    depth = a_task(name="depth", output_topology=OutputTopology.DENSE, objective=Objective.CONTINUOUS)

    with caplog.at_level(logging.INFO):
        built = build_annotators([depth])

    assert built == {}
    assert any("depth" in record.message for record in caplog.records)


def test_a_stacked_view_task_is_skipped_at_the_objective_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A contrastive task predicts an alignment between inputs. Since the axes split,
    its output topology is plain GLOBAL — so the skip now happens one gate later, at
    the objective: METRIC has no per-sample label to show, and the task is named."""
    clip = a_task(
        name="pairs",
        output_topology=OutputTopology.GLOBAL,
        input_topology=InputTopology.MULTISTREAM,
        objective=Objective.METRIC,
    )

    with caplog.at_level(logging.INFO):
        built = build_annotators([clip])

    assert built == {}
    assert any("pairs" in record.message for record in caplog.records)


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
    """One objective produces both readings, so a mixed pair is a wiring bug — named,
    not silently drawn. Uncovered until a mutation check showed nothing guarded it."""
    from src.visualization.annotators import ClassReading, GlobalAnnotation, ValueReading

    truth = ClassReading(presences=(), singular=True)
    predicted = ValueReading(values=np.zeros(1))

    with pytest.raises(TypeError, match="ClassReading.*ValueReading.*one objective"):
        GlobalAnnotation().annotate(
            SampleView(), task_of(OutputTopology.GLOBAL, Objective.MULTICLASS), truth, predicted
        )


def test_a_reading_kind_the_router_does_not_know_is_refused_by_name() -> None:
    """A new Reading member must not reach a run before ``annotate`` can route it.

    The union is the one place a kind exists; the ``match`` in ``annotate`` is the
    one place it is routed. This pins their agreement from the routing side, as
    the retired LABELLERS table test pinned it from the table side.
    """
    from dataclasses import dataclass

    from src.visualization.annotators import GlobalAnnotation

    @dataclass
    class HeatReading:  # a stand-in for a future Reading member
        values: np.ndarray

    pair = HeatReading(values=np.zeros(1))
    with pytest.raises(TypeError, match="HeatReading"):
        task = task_of(OutputTopology.GLOBAL, Objective.CONTINUOUS)
        GlobalAnnotation().annotate(SampleView(), task, pair, pair)  # type: ignore[arg-type]


def test_a_topology_draws_what_it_overrides_and_nothing_else() -> None:
    """`draws` is derived, so it cannot claim a pairing the topology has no branch for.

    Declared beside the implementation it could say yes where nothing was written,
    and the run learned that an epoch in.
    """
    from src.visualization.annotators import (
        ClassReading,
        DenseAnnotation,
        GlobalAnnotation,
        ValueReading,
    )

    assert GlobalAnnotation().draws(ClassReading) is True
    assert GlobalAnnotation().draws(ValueReading) is True
    assert DenseAnnotation().draws(ClassReading) is True
    assert DenseAnnotation().draws(ValueReading) is False  # a heatmap has no label kind yet


def test_a_new_topology_needs_no_change_to_any_reading_kind() -> None:
    """The point of the split: one class, drawing one kind, touching nothing else."""
    from src.visualization.annotators import AnnotationTopology, ClassReading, ValueReading

    class DrawsNumbersOnly(AnnotationTopology):
        def label_values(self, view: SampleView, task: Task, truth: Any, predicted: Any) -> None: ...

    assert DrawsNumbersOnly().draws(ValueReading) is True
    assert DrawsNumbersOnly().draws(ClassReading) is False


def test_a_pairing_a_topology_cannot_draw_names_itself() -> None:
    """Unreachable if `draws` is right — and loud rather than silent if it ever is not."""
    from src.visualization.annotators import DenseAnnotation, ValueReading

    task = task_of(OutputTopology.DENSE, Objective.CONTINUOUS)
    field = ValueReading(values=np.zeros((2, 2)))

    with pytest.raises(TypeError, match="DenseAnnotation has no label for a ValueReading"):
        DenseAnnotation().annotate(SampleView(), task, field, field)
