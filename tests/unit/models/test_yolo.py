"""YOLO facade: offline arch build, loss on a synthetic batch, NMS decode format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from src.models.yolo import build_yolo_model, compute_detection_loss, decode_predictions, normalize_batch_images


def _synthetic_batch(batch_size: int = 2, image_size: int = 64) -> dict[str, torch.Tensor]:
    """Ultralytics-format batch: two images, one centered box each (normalized cxcywh)."""
    return {
        "img": torch.rand(batch_size, 3, image_size, image_size),
        "cls": torch.zeros(batch_size, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]] * batch_size),
        "batch_idx": torch.arange(batch_size, dtype=torch.float32),
    }


class TestNormalizeBatchImages:
    def test_uint8_scaled_to_unit_float(self) -> None:
        """The YOLO dataloader yields uint8 images; the facade owns ultralytics' /255 convention."""
        batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}
        normalized = normalize_batch_images(batch)
        assert normalized["img"].dtype == torch.float32
        assert torch.allclose(normalized["img"], torch.ones(1, 3, 4, 4))

    def test_float_images_pass_through_untouched(self) -> None:
        image = torch.rand(1, 3, 4, 4)
        assert normalize_batch_images({"img": image})["img"] is image


class TestBuildYoloModel:
    def test_builds_offline_from_architecture_yaml(self) -> None:
        model = build_yolo_model("yolov8n.yaml", num_classes=2)
        assert isinstance(model, torch.nn.Module)
        assert model.args is not None  # the loss reads its gains from here

    def test_hyperparameters_reach_model_args(self) -> None:
        model = build_yolo_model("yolov8n.yaml", num_classes=2, hyperparameters={"box": 3.0})
        assert float(cast("Any", model).args.box) == 3.0

    def test_weights_path_loads_trainable(self, tmp_path: Path) -> None:
        """``YOLO()`` loads ``.pt`` weights inference-frozen; the facade must return a trainable model."""
        from ultralytics.nn.tasks import DetectionModel

        weights_path = tmp_path / "tiny.pt"
        torch.save({"model": DetectionModel(cfg="yolov8n.yaml", nc=2, verbose=False)}, weights_path)
        model = build_yolo_model(str(weights_path), num_classes=2)
        assert all(parameter.requires_grad for parameter in model.parameters())


class TestComputeDetectionLoss:
    def test_scalar_total_and_named_components(self) -> None:
        model = build_yolo_model("yolov8n.yaml", num_classes=2)
        total, components = compute_detection_loss(model, _synthetic_batch())
        assert total.ndim == 0  # scalar, ready for Lightning backprop
        assert total.requires_grad
        assert torch.isfinite(total)
        assert set(components) == {"box", "cls", "dfl"}
        assert all(not value.requires_grad for value in components.values())


class TestGroundTruthBoxes:
    def test_torchmetrics_target_format(self) -> None:
        from src.models.yolo import ground_truth_boxes

        targets = ground_truth_boxes(_synthetic_batch(image_size=64), image_size=64)
        assert len(targets) == 2
        for target in targets:
            assert set(target) == {"boxes", "labels"}
            assert target["boxes"].shape == (1, 4)
            assert target["labels"].dtype == torch.int64
        # cxcywh (0.5, 0.5, 0.4, 0.4) at 64px -> xyxy pixels (19.2, 19.2, 44.8, 44.8)
        assert torch.allclose(targets[0]["boxes"][0], torch.tensor([19.2, 19.2, 44.8, 44.8]), atol=1e-4)


class TestComputeLossWithPrecomputedForward:
    def test_precomputed_predictions_reused(self) -> None:
        """The val step runs one forward and feeds it to both the loss and the decoder."""
        model = build_yolo_model("yolov8n.yaml", num_classes=2)
        model.eval()
        batch = _synthetic_batch()
        with torch.no_grad():
            output = model(batch["img"])
            total, components = compute_detection_loss(model, batch, predictions=output)
        assert torch.isfinite(total)
        assert set(components) == {"box", "cls", "dfl"}


class TestDecodePredictions:
    def test_torchmetrics_format(self) -> None:
        model = build_yolo_model("yolov8n.yaml", num_classes=2)
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(2, 3, 64, 64))
        decoded = decode_predictions(output)
        assert len(decoded) == 2
        for prediction in decoded:
            assert set(prediction) == {"boxes", "scores", "labels"}
            assert prediction["boxes"].shape[-1] == 4 or prediction["boxes"].numel() == 0
            assert prediction["labels"].dtype == torch.int64
