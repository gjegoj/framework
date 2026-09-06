"""A composed detection model: ultralytics' graph, our tasks, one backbone for boxes and a class."""

from __future__ import annotations

import pytest
import torch

from src.assembly.models import build_model
from src.core import Batch, DataProfile
from src.core.entities import TargetFacts
from src.models import CompositeModel
from src.models.heads import DetectHead, LinearHead
from tests.support.configs import paper_config

ULTRALYTICS = {"name": "ultralytics", "model_name": "yolov8n.yaml"}
TASKS = {
    "boxes": {"preset": "detection", "target": "objects", "classes": {0: "cat", 1: "dog"}},
    "species": {
        "preset": "classification",
        "target": "species",
        "classes": {0: "cat", 1: "dog"},
        "streams": "features",
    },
}


def profiled() -> DataProfile:
    profile = DataProfile()
    for task in TASKS:
        profile.record(task, TargetFacts(num_classes=2, class_names=["cat", "dog"]))
    return profile


def test_boxes_and_a_whole_image_class_share_one_ultralytics_backbone() -> None:
    model, tasks = build_model(paper_config(model=ULTRALYTICS, tasks=TASKS), profiled())

    assert isinstance(model, CompositeModel)
    assert isinstance(model.heads["boxes"], DetectHead)
    assert isinstance(model.heads["species"], LinearHead)
    assert {task.name for task in tasks} == {"boxes", "species"}


def test_predict_yields_the_raw_detection_tensor_beside_the_class_logits() -> None:
    model, _ = build_model(paper_config(model=ULTRALYTICS, tasks=TASKS), profiled())

    prediction = model.predict(Batch(inputs={"image": torch.zeros(2, 3, 64, 64)}, targets={}))

    assert prediction.logits is not None
    assert tuple(prediction.logits["boxes"].shape) == (2, 4 * 16 + 2, 84)
    assert tuple(prediction.logits["species"].shape) == (2, 2)


def test_a_backbone_without_a_pyramid_is_refused_naming_it() -> None:
    with pytest.raises(LookupError, match="TimmBackbone declares no pyramid"):
        build_model(paper_config(tasks={"boxes": TASKS["boxes"]}), profiled())
