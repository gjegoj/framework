"""Unit tests for the visualization service (IR, annotators, renderer)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.entities import Task, TaskStepView
    from src.visualization.entities import SampleView


class TestVisualizationEntities:
    def test_classification_carries_label_and_confidence(self) -> None:
        from src.visualization.entities import Classification

        label = Classification(label="cat", confidence=0.91)
        assert label.label == "cat"
        assert label.confidence == 0.91

    def test_classification_confidence_optional(self) -> None:
        from src.visualization.entities import Classification

        assert Classification(label="cat").confidence is None

    def test_classifications_holds_items(self) -> None:
        from src.visualization.entities import Classification, Classifications

        multi = Classifications(items=[Classification("a"), Classification("b", 0.5)])
        assert [c.label for c in multi.items] == ["a", "b"]

    def test_sample_view_defaults(self) -> None:
        from src.visualization.entities import SampleView

        sample = SampleView(image=np.zeros((4, 4, 3), dtype=np.uint8))
        assert sample.image.shape == (4, 4, 3)
        assert sample.fields == {}
        assert sample.tags == []
        assert sample.metadata == {}


class TestAnnotators:
    def _task(self, objective: str = "multiclass") -> "Task":
        from dataclasses import replace

        from src.core.entities import Task
        from src.tasks.presets import classification

        names = ["cat", "cow", "dog"]
        task = replace(classification("species", num_classes=3, objective=objective), class_names=names)
        assert isinstance(task, Task)
        return task

    def _view(self, preds: list[list[float]], target: list[object]) -> "TaskStepView":
        import torch

        from src.core.entities import TaskStepView

        return TaskStepView(preds=torch.tensor(preds), metric_target=torch.tensor(target))

    def test_registry_keyed_by_axes(self) -> None:
        from src.tasks.taxonomy import Objective, Topology
        from src.visualization.annotators import annotators, axes_key

        assert axes_key(Topology.GLOBAL, Objective.MULTICLASS) in annotators
        assert axes_key(Topology.GLOBAL, Objective.MULTILABEL) in annotators

    def test_classification_annotator_adds_gt_pred_and_tag(self) -> None:
        import numpy as np

        from src.visualization.annotators import ClassificationAnnotator
        from src.visualization.entities import Classification, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        view = self._view(preds=[[0.1, 0.2, 0.7]], target=[2])  # softmax-like, pred=dog, gt=dog
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
        view = self._view(preds=[[0.8, 0.1, 0.1]], target=[2])  # pred=cat, gt=dog
        ClassificationAnnotator().annotate(sample, self._task(), view, index=0)
        assert "species:wrong" in sample.tags

    def test_multilabel_annotator_builds_classifications(self) -> None:
        import numpy as np

        from src.visualization.annotators import MultilabelAnnotator
        from src.visualization.entities import Classifications, SampleView

        sample = SampleView(image=np.zeros((2, 2, 3), dtype=np.uint8))
        # sigmoid preds: cat=0.9, cow=0.1, dog=0.8 ; gt multi-hot: cat=1, dog=1
        view = self._view(preds=[[0.9, 0.1, 0.8]], target=[[1.0, 0.0, 1.0]])
        MultilabelAnnotator(threshold=0.5).annotate(sample, self._task("multilabel"), view, index=0)

        gt = sample.fields["species_gt"]
        pred = sample.fields["species_pred"]
        assert isinstance(gt, Classifications) and {c.label for c in gt.items} == {"cat", "dog"}
        assert isinstance(pred, Classifications) and {c.label for c in pred.items} == {"cat", "dog"}


class TestPipeline:
    def test_build_sample_views_annotates_each_image(self) -> None:
        import numpy as np

        from src.visualization.entities import Classification
        from src.visualization.pipeline import build_sample_views

        annotator_tests = TestAnnotators()
        task = annotator_tests._task()
        views = {
            "species": annotator_tests._view(
                preds=[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
                target=[2, 2],
            )
        }
        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        samples = build_sample_views(images, [task], views)

        assert len(samples) == 2
        gt = samples[0].fields["species_gt"]
        pred = samples[0].fields["species_pred"]
        assert isinstance(gt, Classification) and gt.label == "dog"
        assert isinstance(pred, Classification)
        assert samples[1].tags == ["species:wrong"]


class TestRenderer:
    def _sample(self) -> "SampleView":
        import numpy as np

        from src.visualization.entities import Classification, SampleView

        return SampleView(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            fields={
                "species_gt": Classification("cat"),
                "species_pred": Classification("dog", confidence=0.73),
            },
            tags=["species:wrong"],
        )

    def test_label_renderer_registry_by_type(self) -> None:
        from src.visualization.entities import Classification, Classifications
        from src.visualization.renderer import label_renderers

        assert Classification.__name__ in label_renderers
        assert Classifications.__name__ in label_renderers

    def test_classification_caption(self) -> None:
        from src.visualization.entities import Classification
        from src.visualization.renderer import ClassificationLabelRenderer

        assert ClassificationLabelRenderer().caption(Classification("dog", 0.73)) == "dog (0.73)"
        assert ClassificationLabelRenderer().caption(Classification("dog")) == "dog"

    def test_classifications_caption(self) -> None:
        from src.visualization.entities import Classification, Classifications
        from src.visualization.renderer import ClassificationsLabelRenderer

        text = ClassificationsLabelRenderer().caption(Classifications([Classification("a"), Classification("b", 0.5)]))
        assert "a" in text and "b" in text

    def test_plotly_renderer_returns_html_with_labels(self) -> None:
        from src.visualization.renderer import PlotlyRenderer

        html = PlotlyRenderer().render([self._sample()], title="samples/val")
        assert isinstance(html, str)
        assert "cat" in html and "dog" in html  # gt + pred captions present
        assert "plotly" in html.lower()  # plotly div embedded

    def test_plotly_renderer_handles_empty(self) -> None:
        from src.visualization.renderer import PlotlyRenderer

        assert isinstance(PlotlyRenderer().render([], title="empty"), str)

    def test_plotly_renderer_multitask_grid_sets_explicit_height(self) -> None:
        import numpy as np

        from src.visualization.entities import Classification, Classifications, SampleView
        from src.visualization.renderer import PlotlyRenderer

        samples = [
            SampleView(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                fields={
                    "species_gt": Classification("cat"),
                    "species_pred": Classification("dog", confidence=0.71),
                    "is_cat_gt": Classification("not_cat"),
                    "is_cat_pred": Classification("cat", confidence=0.62),
                    "tags_gt": Classifications([Classification("indoor"), Classification("small")]),
                    "tags_pred": Classifications(
                        [Classification("indoor", 0.9), Classification("small", 0.6), Classification("fluffy", 0.55)]
                    ),
                },
                tags=["species:wrong", "is_cat:wrong", "tags:wrong"],
            )
            for _ in range(4)
        ]
        html = PlotlyRenderer().render(samples, title="samples/val")
        assert '"height":' in html
        assert "species: gt=cat" in html
        assert "tags: gt=" in html
