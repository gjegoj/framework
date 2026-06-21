"""Pipeline: batch tensors → ``SampleView`` IR via per-task annotators."""

from __future__ import annotations

import numpy as np

from src.core.entities import Task, TaskStepView
from src.visualization.annotators import annotators, axes_key
from src.visualization.entities import SampleView


def build_sample_views(
    images: np.ndarray,
    tasks: list[Task],
    task_views: dict[str, TaskStepView],
    sources: list[str] | None = None,
) -> list[SampleView]:
    """Build one ``SampleView`` per image, annotated by every supported task.

    Parameters:
        images (np.ndarray): Display-ready ``[N, H, W, C]`` uint8 RGB batch.
        tasks (list[Task]): Active tasks (topology/objective select the annotator).
        task_views (dict[str, TaskStepView]): Per-task step views from ``StepOutput``.
        sources (list[str] | None): Per-sample source path/URL of the displayed image
            (batch order), surfaced as an "open original" link. ``None`` to omit.

    Returns:
        list[SampleView]: Annotated samples in batch order.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be [N, H, W, C], got shape {images.shape}.")

    samples: list[SampleView] = []
    for index in range(images.shape[0]):
        source = sources[index] if sources is not None and index < len(sources) else None
        sample = SampleView(image=images[index], source=source)
        for task in tasks:
            view = task_views.get(task.name)
            if view is None:
                continue
            key = axes_key(task.topology, task.objective)
            if key not in annotators:
                continue
            annotators.create(key).annotate(sample, task, view, index)
        samples.append(sample)
    return samples
