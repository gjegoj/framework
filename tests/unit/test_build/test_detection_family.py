"""A composed detection model: ultralytics' graph, our tasks, one backbone for boxes and a class."""

from __future__ import annotations

import pytest
import torch

from src.core import Batch
from src.core.entities import TaskFacts
from src.models import CompositeModel
from src.models.heads import DetectHead, LinearHead
from tests.support.configs import model_of, paper_config

ULTRALYTICS = {"name": "ultralytics", "model_name": "yolov8n.yaml"}
TASKS = {
    "boxes": {"kind": "detection", "target": "objects", "classes": {0: "cat", 1: "dog"}},
    "species": {
        "kind": "classification",
        "target": "species",
        "classes": {0: "cat", 1: "dog"},
        "streams": "features",
    },
}


FACTS = {task: TaskFacts(num_classes=2, class_names=("cat", "dog")) for task in TASKS}
"""What setup would have learned: two classes, on both tasks."""


def test_boxes_and_a_whole_image_class_share_one_ultralytics_backbone() -> None:
    model, tasks = model_of(paper_config(model=ULTRALYTICS, tasks=TASKS), FACTS)

    assert isinstance(model, CompositeModel)
    assert isinstance(model.heads["boxes"], DetectHead)
    assert isinstance(model.heads["species"], LinearHead)
    assert {task.name for task in tasks} == {"boxes", "species"}


def test_predict_yields_the_raw_detection_tensor_beside_the_class_logits() -> None:
    model, _ = model_of(paper_config(model=ULTRALYTICS, tasks=TASKS), FACTS)

    prediction = model.predict(Batch(inputs={"image": torch.zeros(2, 3, 64, 64)}, targets={}))

    assert prediction.logits is not None
    assert tuple(prediction.logits["boxes"].shape) == (2, 4 * 16 + 2, 84)
    assert tuple(prediction.logits["species"].shape) == (2, 2)


def test_a_backbone_without_a_pyramid_is_refused_naming_it() -> None:
    with pytest.raises(LookupError, match="TimmBackbone declares no pyramid"):
        model_of(paper_config(tasks={"boxes": TASKS["boxes"]}), FACTS)
