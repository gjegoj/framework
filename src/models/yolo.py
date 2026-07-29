"""YOLO facade over ultralytics: model building, detection loss, prediction decoding.

ultralytics is a detail of this module (and of the detection data module, which needs
its dataset builders). License note: ultralytics is AGPL-3.0 — commercial use is the
project owner's licensing decision, acknowledged in the detection design spec.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from src.core.entities import LossResult
from src.models.complete import CompleteModel
from src.models.registry import register_complete_model


def build_yolo_model(model: str, num_classes: int, hyperparameters: dict[str, Any] | None = None) -> nn.Module:
    """Build a YOLO ``DetectionModel`` from a ``.yaml`` architecture or ``.pt`` weights.

    The returned module carries ``args`` (ultralytics defaults merged with
    ``hyperparameters``) — its own criterion reads the loss gains (``box``/``cls``/
    ``dfl``) from there.

    Parameters:
        model (str): Architecture yaml (offline, e.g. ``yolov8n.yaml``) or weights path (``.pt``).
        num_classes (int): Detection class count (overrides the architecture default).
        hyperparameters (dict[str, Any] | None): ultralytics hyp overrides, forwarded verbatim.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    if model.endswith(".pt"):
        from ultralytics import YOLO

        detection_model = cast("nn.Module", YOLO(model).model)
        # Loaded weights come inference-frozen; ultralytics' own trainer re-enables
        # gradients in its setup, which our Lightning contour replaces.
        detection_model.requires_grad_(True)
    else:
        detection_model = DetectionModel(cfg=model, nc=num_classes, verbose=False)
    # ``args`` is ultralytics' dynamic hyp namespace — invisible to nn.Module typing.
    cast("Any", detection_model).args = get_cfg(overrides=dict(hyperparameters or {}))
    return detection_model


def normalize_batch_images(batch: dict[str, Any]) -> dict[str, Any]:
    """Scale a batch's uint8 images to float ``[0, 1]``, as ultralytics' trainer does.

    The YOLO dataloader yields uint8 ``img`` tensors; ultralytics converts them in its
    trainer's ``preprocess_batch`` (``.float() / 255``), which our Lightning contour
    replaces — so the facade owns the convention. Float images pass through untouched.

    Parameters:
        batch (dict[str, Any]): Ultralytics batch (``img``/``cls``/``bboxes``/``batch_idx``).

    Returns:
        dict[str, Any]: The same batch, with ``img`` scaled in place when it was uint8.
    """
    image = batch["img"]
    if image.dtype == torch.uint8:
        batch["img"] = image.float() / 255
    return batch


def compute_detection_loss(
    model: nn.Module, batch: dict[str, Any], predictions: Any | None = None
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run the model's own criterion on an ultralytics-format batch.

    ultralytics returns a per-component ``[3]`` tensor (already scaled by batch size —
    kept as-is so optimization matches their trainer) plus a dict of detached component
    scalars keyed ``box_loss``/``cls_loss``/``dfl_loss``.

    Parameters:
        model (nn.Module): A model built by :func:`build_yolo_model`.
        batch (dict[str, Any]): Ultralytics batch (``img``/``cls``/``bboxes``/``batch_idx``).
        predictions (Any | None): Precomputed forward output to reuse (the validation
            step runs one forward and feeds it to both the loss and the decoder).

    Returns:
        tuple[Tensor, dict[str, Tensor]]: Scalar total for backprop, and the detached
        components renamed to ``box``/``cls``/``dfl``.
    """
    total, items = cast("Any", model).loss(batch, preds=predictions)  # ultralytics API, invisible to typing
    components = {name.removesuffix("_loss"): value for name, value in items.items()}
    return total.sum(), components


def ground_truth_boxes(batch: dict[str, Any], image_size: int) -> list[dict[str, Tensor]]:
    """Convert an ultralytics batch's targets to the torchmetrics detection format.

    ``bboxes`` are normalized cxcywh over the (letterboxed) ``image_size`` square;
    ``batch_idx`` maps each box to its image.

    Parameters:
        batch (dict[str, Any]): Ultralytics batch (``img``/``cls``/``bboxes``/``batch_idx``).
        image_size (int): Square image size the normalized boxes refer to.

    Returns:
        list[dict[str, Tensor]]: Per image: ``boxes`` (xyxy pixels) and ``labels``.
    """
    from ultralytics.utils.ops import xywh2xyxy

    boxes = xywh2xyxy(batch["bboxes"]) * image_size
    labels = batch["cls"].reshape(-1).to(torch.int64)
    image_indices = batch["batch_idx"].to(torch.int64)
    batch_size = int(batch["img"].shape[0])
    targets: list[dict[str, Tensor]] = []
    for image_index in range(batch_size):
        selection = image_indices == image_index
        targets.append({"boxes": boxes[selection], "labels": labels[selection]})
    return targets


def decode_predictions(
    model_output: Any, confidence_threshold: float = 0.25, iou_threshold: float = 0.7
) -> list[dict[str, Tensor]]:
    """NMS-decode raw eval-mode output into the torchmetrics detection format.

    Parameters:
        model_output (Any): Eval-mode forward output (tuple whose first element is the
            concatenated ``[B, 4 + num_classes, anchors]`` prediction tensor).
        confidence_threshold (float): Minimum detection confidence kept.
        iou_threshold (float): NMS IoU threshold.

    Returns:
        list[dict[str, Tensor]]: Per image: ``boxes`` (xyxy), ``scores``, ``labels``.
    """
    from ultralytics.utils.nms import non_max_suppression

    raw = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
    detections = non_max_suppression(raw, conf_thres=confidence_threshold, iou_thres=iou_threshold)
    return [
        {"boxes": boxes[:, :4], "scores": boxes[:, 4], "labels": boxes[:, 5].to(torch.int64)} for boxes in detections
    ]


DetectionPredictions = list[dict[str, Tensor]]
DetectionTargets = list[dict[str, Tensor]]


@register_complete_model("yolo")
class YoloModel(CompleteModel[DetectionPredictions, DetectionTargets]):
    """ultralytics YOLO as a complete model: head and loss fused, framework drives the loop.

    Thin adapter over this module's facade functions; the wrapped ``DetectionModel``
    lives as a submodule so device moves / EMA / checkpoints see its parameters.

    Parameters:
        name (str): Architecture yaml (offline, e.g. ``yolov8n.yaml``) or ``.pt`` weights path.
        num_classes (int): Detection class count (from the dataset descriptor).
        image_size (int): Square image size — converts normalized targets for metrics.
        hyperparameters (dict[str, Any] | None): ultralytics hyp overrides, forwarded verbatim.
        confidence_threshold (float): Minimum detection confidence kept at evaluation.
        iou_threshold (float): NMS IoU threshold at evaluation.
    """

    family = "detection"

    def __init__(
        self,
        name: str,
        num_classes: int,
        image_size: int,
        hyperparameters: dict[str, Any] | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self._detection_model = build_yolo_model(name, num_classes=num_classes, hyperparameters=hyperparameters)
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return normalize_batch_images(batch)

    def forward(self, batch: dict[str, Any]) -> Any:
        return self._detection_model(batch["img"])

    def training_loss(self, batch: dict[str, Any]) -> LossResult:
        total, components = compute_detection_loss(self._detection_model, batch)
        return LossResult(total=total, components=components)

    def evaluation_loss(self, batch: dict[str, Any], output: Any) -> LossResult:
        total, components = compute_detection_loss(self._detection_model, batch, predictions=output)
        return LossResult(total=total, components=components)

    def predictions(self, output: Any) -> DetectionPredictions:
        return decode_predictions(output, self._confidence_threshold, self._iou_threshold)

    def targets(self, batch: dict[str, Any]) -> DetectionTargets:
        return ground_truth_boxes(batch, self._image_size)
