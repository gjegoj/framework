"""Shared pure builders for tests — plain functions returning fresh objects, no pytest fixtures.

These replace the former cross-test-module imports (``tests.test_config._raw``,
``tests.test_composition._minimal_config``, ``tests.test_data._make_transform``): tests may
import shared infrastructure only from ``tests.support``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.core.entities import Task, TaskStepView
from src.tasks.presets import task_presets
from src.transforms.sample import AlbumentationsTransform

RESIZE_NORMALIZE_TOTENSOR = [
    {"_target_": "albumentations.Resize", "height": 64, "width": 64},
    {"_target_": "albumentations.Normalize"},
    {"_target_": "albumentations.pytorch.ToTensorV2"},
]

MINIMAL_TRANSFORMS: dict[str, Any] = {
    "train": {"_target_": "albumentations.Compose", "transforms": RESIZE_NORMALIZE_TOTENSOR},
    "val": {"_target_": "albumentations.Compose", "transforms": RESIZE_NORMALIZE_TOTENSOR},
}


def raw_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid raw config, with optional top-level overrides."""
    base: dict[str, Any] = {
        "project": "demo",
        "epochs": 5,
        "batch_size": 8,
        "lr": 1e-3,
        "image_size": [224, 224],
        "data": {
            "sources": "data/classification.csv",
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {
            "label": {"preset": "classification", "target": "label", "class_mapping": {0: "cat", 1: "cow", 2: "dog"}}
        },
    }
    base.update(overrides)
    return base


def minimal_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal raw config with inline transforms (the composition-wiring flavour)."""
    base: dict[str, Any] = {
        "project": "test",
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "image_size": [64, 64],
        "data": {
            "sources": "data.csv",
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {
            "label": {"preset": "classification", "target": "label", "class_mapping": {0: "cat", 1: "cow", 2: "dog"}}
        },
        "transforms": MINIMAL_TRANSFORMS,
    }
    base.update(overrides)
    return base


def make_transform(
    size: tuple[int, int] = (16, 16),
    spatial: list[str] | None = None,
) -> AlbumentationsTransform:
    """Resize/Normalize/ToTensor pipeline sized to ``size``; ``spatial`` registers mask targets."""
    height, width = size
    compose = A.Compose([A.Resize(height, width, mask_interpolation=0), A.Normalize(), ToTensorV2()])
    return AlbumentationsTransform(compose, spatial_targets=spatial)


def make_task(
    preset_name: str,
    task_name: str,
    num_classes: int,
    objective: str | None = None,
    class_names: list[str] | None = None,
) -> Task:
    """Build a task from a preset for visualization/pipeline tests (optionally with class names)."""
    task = task_presets.create(preset_name)(task_name, num_classes=num_classes, objective=objective)
    return dataclasses.replace(task, class_names=class_names) if class_names is not None else task


def make_view(predictions: Tensor | list[Any], target: Tensor | list[Any]) -> TaskStepView:
    """Build a TaskStepView; plain lists are tensorised, ready tensors pass through unchanged."""
    prediction_tensor = torch.tensor(predictions) if isinstance(predictions, list) else predictions
    target_tensor = torch.tensor(target) if isinstance(target, list) else target
    return TaskStepView(predictions=prediction_tensor, metric_target=target_tensor)
