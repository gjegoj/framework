"""Annotators: per-(topology, objective) GT-vs-prediction field writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests.support.builders import make_task, make_view

if TYPE_CHECKING:
    from src.core.entities import Task, TaskStepView


class TestAnnotators:
    def _task(self, objective: str = "multiclass") -> "Task":
        return make_task("classification", "species", 3, objective=objective, class_names=["cat", "cow", "dog"])

    def _view(self, predictions: list[list[float]], target: list[object]) -> "TaskStepView":
        return make_view(predictions, target)

    def test_registry_keyed_by_axes(self) -> None:
        from src.core.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators

        assert (Topology.GLOBAL, Objective.MULTICLASS) in annotators
        assert (Topology.GLOBAL, Objective.MULTILABEL) in annotators

    def test_classification_annotator_adds_gt_pred_and_tag(self) -> None:
        import numpy as np

        from src.visualization.annotators import ClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(predictions=[[0.1, 0.2, 0.7]], target=[2])  # softmax-like, pred=dog, gt=dog
        ClassificationAnnotator().annotate(sample, self._task(), view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification) and pred.label == "dog"
        assert abs((pred.confidence or 0.0) - 0.7) < 1e-6
        assert "species:correct" in sample.tags

    def test_classification_annotator_wrong_tag(self) -> None:
        import numpy as np

        from src.visualization.annotators import ClassificationAnnotator
        from src.visualization.entities import SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(predictions=[[0.8, 0.1, 0.1]], target=[2])  # pred=cat, gt=dog
        ClassificationAnnotator().annotate(sample, self._task(), view, index=0)
        assert "species:wrong" in sample.tags

    def test_binary_annotator_thresholds_positive_class(self) -> None:
        from dataclasses import replace

        import numpy as np

        from src.tasks.presets import classification
        from src.visualization.annotators import BinaryClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        task = replace(classification("species", num_classes=2, objective="binary"), class_names=["cat", "dog"])
        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # sigmoid P(positive=dog) = 0.8 → pred dog; gt = 1 (dog)
        view = self._view(predictions=[[0.8]], target=[[1]])
        BinaryClassificationAnnotator().annotate(sample, task, view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification) and pred.label == "dog"
        assert abs((pred.confidence or 0.0) - 0.8) < 1e-6
        assert "species:correct" in sample.tags

    def test_binary_annotator_below_threshold_is_negative_class(self) -> None:
        from dataclasses import replace

        import numpy as np

        from src.tasks.presets import classification
        from src.visualization.annotators import BinaryClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        task = replace(classification("species", num_classes=2, objective="binary"), class_names=["cat", "dog"])
        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # P(dog) = 0.3 → pred cat with confidence 0.7; gt = 1 (dog) → wrong
        view = self._view(predictions=[[0.3]], target=[[1]])
        BinaryClassificationAnnotator().annotate(sample, task, view, index=0)

        pred = sample.fields["species_pred"]
        assert isinstance(pred, Classification) and pred.label == "cat"
        assert abs((pred.confidence or 0.0) - 0.7) < 1e-6
        assert "species:wrong" in sample.tags

    def test_binary_registered_for_global_binary(self) -> None:
        from src.core.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators

        assert (Topology.GLOBAL, Objective.BINARY) in annotators

    def test_multilabel_annotator_builds_classifications(self) -> None:
        import numpy as np

        from src.visualization.annotators import MultilabelAnnotator
        from src.visualization.entities import Classifications, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # sigmoid predictions: cat=0.9, cow=0.1, dog=0.8 ; gt multi-hot: cat=1, dog=1
        view = self._view(predictions=[[0.9, 0.1, 0.8]], target=[[1.0, 0.0, 1.0]])
        MultilabelAnnotator(threshold=0.5).annotate(sample, self._task("multilabel"), view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classifications) and {c.label for c in gt.items} == {"cat", "dog"}
        assert isinstance(pred, Classifications) and {c.label for c in pred.items} == {"cat", "dog"}


class TestRegressionAnnotator:
    def _task(self) -> "Task":
        return make_task("regression", "age", 1)

    def _view(self, predictions: list[list[float]], target: list[list[float]]) -> "TaskStepView":
        return make_view(predictions, target)

    def test_registered_for_global_continuous(self) -> None:
        from src.core.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators

        assert (Topology.GLOBAL, Objective.CONTINUOUS) in annotators

    def test_scalar_gt_pred_and_signed_error(self) -> None:
        import numpy as np

        from src.visualization.annotators import RegressionAnnotator
        from src.visualization.entities import Regression, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(predictions=[[3.51]], target=[[3.42]])
        RegressionAnnotator().annotate(sample, self._task(), view, index=0)

        gt = sample.fields["age_gt"]
        pred = sample.fields["age_pred"]
        assert isinstance(gt, Regression) and isinstance(pred, Regression)
        assert gt.components[0].name == "" and abs(gt.components[0].value - 3.42) < 1e-5
        assert gt.components[0].error is None
        assert abs(pred.components[0].value - 3.51) < 1e-5
        assert abs((pred.components[0].error or 0.0) - 0.09) < 1e-5  # signed pred - gt


class TestSegmentationAnnotator:
    def _task(self) -> "Task":
        return make_task("segmentation", "mask", 3)

    def _view(self, predictions: object, target: object) -> "TaskStepView":
        return make_view(predictions, target)  # type: ignore[arg-type]

    def test_registered_for_dense_multiclass(self) -> None:
        from src.core.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators

        assert (Topology.DENSE, Objective.MULTICLASS) in annotators

    def test_builds_per_class_masks_from_logits_and_label_map(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import SegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        # predictions [B, C=3, H=4, W=4] softmax-ish; argmax over C gives the pred label map
        probs = torch.zeros(1, 3, 4, 4)
        probs[0, 0, :2, :] = 0.9  # top half class 0
        probs[0, 1, 2:, :] = 0.9  # bottom half class 1
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 2:, :] = 1  # gt: bottom half class 1

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        SegmentationAnnotator().annotate(sample, self._task(), self._view(probs, target), index=0)

        gt = sample.fields["mask_gt"]
        pred = sample.fields["mask_pred"]
        assert isinstance(gt, Segmentation) and isinstance(pred, Segmentation)
        gt_classes = {c.name for c in gt.classes}
        assert gt_classes == {"0", "1"}  # both present in gt
        road = next(c for c in gt.classes if c.name == "1")
        assert road.mask.shape == (4, 4) and bool(road.mask[3, 0])  # bottom half is class 1

    def test_ignore_index_skips_class(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import SegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        probs = torch.zeros(1, 3, 4, 4)
        probs[0, 0, :2, :] = 0.9
        probs[0, 1, 2:, :] = 0.9
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 2:, :] = 1

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        SegmentationAnnotator(ignore_index=0).annotate(sample, self._task(), self._view(probs, target), index=0)

        gt = sample.fields["mask_gt"]
        assert isinstance(gt, Segmentation)
        assert {c.name for c in gt.classes} == {"1"}  # class 0 (background) skipped


class TestMultilabelSegmentationAnnotator:
    def _task(self) -> "Task":
        return make_task("segmentation", "mask", 2, objective="multilabel")

    def _view(self, predictions: object, target: object) -> "TaskStepView":
        return make_view(predictions, target)  # type: ignore[arg-type]

    def test_registered_for_dense_multilabel(self) -> None:
        from src.core.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators

        assert (Topology.DENSE, Objective.MULTILABEL) in annotators

    def test_overlapping_masks_from_per_channel_threshold(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import MultilabelSegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        # predictions [B, C=2, H=4, W=4] sigmoid; gt [B, C=2, H=4, W=4] multi-hot (classes overlap)
        probs = torch.zeros(1, 2, 4, 4)
        probs[0, 0] = 0.9  # class 0 active everywhere
        probs[0, 1, :2, :2] = 0.9  # class 1 active only top-left → overlaps class 0
        gt = torch.zeros(1, 2, 4, 4)
        gt[0, 0] = 1.0
        gt[0, 1, :2, :2] = 1.0

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        MultilabelSegmentationAnnotator(threshold=0.5).annotate(sample, self._task(), self._view(probs, gt), index=0)

        pred = sample.fields["mask_pred"]
        assert isinstance(pred, Segmentation)
        masks = {c.name: c.mask for c in pred.classes}
        assert set(masks) == {"0", "1"}
        assert bool(masks["0"][0, 0]) and bool(masks["1"][0, 0])  # same pixel in BOTH classes
        assert bool(masks["0"][3, 3]) and not bool(masks["1"][3, 3])  # class 1 absent bottom-right

    def test_threshold_drops_low_confidence_class(self) -> None:
        import numpy as np
        import torch

        from src.visualization.annotators import MultilabelSegmentationAnnotator
        from src.visualization.entities import SampleView, Segmentation

        probs = torch.zeros(1, 2, 4, 4)
        probs[0, 0] = 0.9  # present
        probs[0, 1] = 0.3  # below threshold everywhere → absent
        gt = torch.zeros(1, 2, 4, 4)
        gt[0, 0] = 1.0

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        MultilabelSegmentationAnnotator(threshold=0.5).annotate(sample, self._task(), self._view(probs, gt), index=0)

        pred = sample.fields["mask_pred"]
        assert isinstance(pred, Segmentation)
        assert {c.name for c in pred.classes} == {"0"}  # class 1 thresholded out


class TestMetricTaskAnnotation:
    def test_global_metric_task_is_skipped_gracefully(self) -> None:
        import dataclasses

        from src.core.taxonomy import Objective, Topology
        from src.visualization.pipeline import build_sample_views

        base = make_task("classification", "species", 3, class_names=["cat", "cow", "dog"])
        task = dataclasses.replace(base, topology=Topology.GLOBAL, objective=Objective.METRIC)
        views = {task.name: make_view([[0.1, 0.2]], [[0.1, 0.2]])}

        samples = build_sample_views(np.zeros((1, 4, 4, 3), np.uint8), [task], views)

        assert samples[0].fields == {}  # no annotator for (GLOBAL, METRIC): task silently skipped
