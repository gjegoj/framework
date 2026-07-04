"""Pipeline: batch tensors -> annotated SampleViews via registered annotators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tests.support.builders import make_task, make_view

if TYPE_CHECKING:
    pass


class TestPipeline:
    def test_build_sample_views_annotates_each_image(self) -> None:

        from src.visualization.entities import Classification
        from src.visualization.pipeline import build_sample_views

        task = make_task("classification", "species", 3, class_names=["cat", "cow", "dog"])
        views = {
            "species": make_view(
                predictions=[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
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

    def test_build_sample_views_threads_sources(self) -> None:

        from src.visualization.pipeline import build_sample_views

        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        samples = build_sample_views(images, [], {}, sources=["a.jpg", "b.jpg"])
        assert [sample.source for sample in samples] == ["a.jpg", "b.jpg"]

    def test_build_sample_views_source_none_by_default(self) -> None:

        from src.visualization.pipeline import build_sample_views

        assert build_sample_views(np.zeros((1, 4, 4, 3), dtype=np.uint8), [], {})[0].source is None


class TestPipelineRegression:
    def test_regression_task_is_annotated_not_skipped(self) -> None:
        import torch

        from src.core.entities import Task, TaskStepView
        from src.tasks.presets import regression
        from src.visualization.entities import Regression
        from src.visualization.pipeline import build_sample_views

        task = regression("age", num_classes=1)
        assert isinstance(task, Task)
        views = {
            "age": TaskStepView(predictions=torch.tensor([[3.51], [7.0]]), metric_target=torch.tensor([[3.42], [5.0]]))
        }
        images = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        samples = build_sample_views(images, [task], views)
        assert isinstance(samples[0].fields["age_gt"], Regression)
        assert isinstance(samples[1].fields["age_pred"], Regression)
