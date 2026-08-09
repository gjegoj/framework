"""``YoloModel``: ultralytics behind the ``Model`` port, translating in both directions."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from src.core import Batch, Instances, Modality, Prediction
from src.core.ports import Model

pytest.importorskip("ultralytics", reason="the YOLO family is an optional dependency")

from src.models import YoloModel

CLASSES = 3


def model() -> YoloModel:
    return YoloModel(model_name="yolov8n.yaml", num_classes=CLASSES)


def batch(images: int = 2) -> Batch:
    """Two images with one object each, in the framework's own currency."""
    return Batch(
        inputs={Modality.IMAGE: torch.rand(images, 3, 64, 64)},
        targets={
            "boxes": Instances(
                boxes=torch.tensor([[8.0, 8.0, 24.0, 24.0]] * images),
                labels=torch.zeros(images, dtype=torch.int64),
                sample_index=torch.arange(images),
            )
        },
    )


def test_it_is_a_model_like_any_other() -> None:
    assert isinstance(model(), Model)


def test_the_vendors_three_losses_arrive_as_named_parts() -> None:
    """box, cls and dfl are ultralytics' own, computed against its own assigner.

    Re-deriving them would be a different loss wearing the same name. Named parts put
    them on the same footing as every other criterion's, so `train/boxes/box` logs like
    `train/label/ce` and nothing downstream learns a second vocabulary.
    """
    result = model().step(batch())

    # Scoped by the task, exactly as a composed model scopes its criteria's parts, so
    # `train/boxes/box` reads like `train/label/ce` and no consumer learns a second grammar.
    assert set(result.loss.parts) == {"boxes/box", "boxes/cls", "boxes/dfl"}
    assert result.loss.total.requires_grad
    assert result.loss.total.ndim == 0


def test_the_loss_is_given_exactly_the_keys_its_criterion_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured: the detection criterion reads batch_idx, cls and bboxes, and nothing else.

    Handed our `Batch` it would fail inside ultralytics at a frame that explains nothing,
    so the dialect is rebuilt here — which is what an adapter is for.
    """
    built = model()
    seen: dict[str, Any] = {}

    def record(vendor_batch: dict[str, Any], preds: Any = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seen.update(vendor_batch)
        return torch.zeros(3, requires_grad=True), {"box_loss": torch.tensor(0.0)}

    monkeypatch.setattr(built.detector, "loss", record)

    built.step(batch())

    assert {"img", "cls", "bboxes", "batch_idx"} <= set(seen)
    assert seen["bboxes"].shape[-1] == 4


def test_nobody_has_to_be_told_the_image_size() -> None:
    """The reference passed `image_size` in so the model could un-normalise boxes for
    metrics — a second copy of a number the data module already has, and two copies can
    disagree with nothing to notice. The image in the batch carries its own shape.
    """
    assert "image_size" not in inspect.signature(YoloModel.__init__).parameters


def test_a_prediction_is_the_objects_left_after_suppression() -> None:
    """A detection prediction is what survived NMS, per image — the framework's ragged
    shape rather than a raw anchor tensor no consumer could read.
    """
    built = model()
    built.eval()

    prediction = built.predict(batch())

    assert isinstance(prediction, Prediction)
    found = prediction.outputs["boxes"]
    assert isinstance(found, Instances)
    assert found.scores is not None
    assert found.boxes.shape[-1] == 4
    assert set(found.sample_index.tolist()) <= {0, 1}


def test_the_step_hands_metrics_the_targets_it_was_given() -> None:
    """The port says the model owns target adaptation, so what reaches a metric is
    ready to compare — and for this family that is the batch's own objects, untouched.
    """
    result = model().step(batch())

    assert isinstance(result.targets["boxes"], Instances)


def test_it_is_filed_under_its_own_name() -> None:
    """A vendor family has no backbone to be named after, and the port's default says so."""
    assert model().architecture == "YoloModel"


def test_a_per_task_rate_against_it_is_refused_rather_than_ignored() -> None:
    """It exposes no per-task parameters, so a declared rate has nothing to move — and the
    module refuses it instead of training at a pace nobody asked for.
    """
    assert list(model().task_parameters("boxes")) == []


def test_a_training_step_hands_back_no_prediction_rather_than_an_empty_one() -> None:
    """A detection head emits feature maps while training and assembles the decodable
    tensor only in eval — ultralytics does not spend the decode where nobody reads it.

    Empty objects would be a fabricated answer, and a train-stage mAP would report it as
    zero: a measurement nobody took, dressed as a bad model.
    """
    built = model()
    built.train()

    result = built.step(batch())

    assert result.prediction.outputs == {}
    assert result.loss.total.requires_grad
