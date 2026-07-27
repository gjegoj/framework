"""DetectionLitModule: loss logging, mAP accumulation, metric directions."""

from __future__ import annotations

import torch

from src.core.entities import LossResult
from src.core.enums import Stage
from src.models.yolo import build_yolo_model
from src.training.modules.detection import DetectionLitModule
from src.training.optim import OptimizerBuilder


def _module() -> DetectionLitModule:
    return DetectionLitModule(
        model=build_yolo_model("yolov8n.yaml", num_classes=2),
        task_name="boxes",
        image_size=64,
        optimizer_builder=OptimizerBuilder(base_lr=1e-3),
    )


def _batch(batch_size: int = 2, image_size: int = 64) -> dict[str, torch.Tensor]:
    """A minimal ultralytics-format batch; ``img`` is uint8 like the real YOLO dataloader yields."""
    return {
        "img": torch.randint(0, 256, (batch_size, 3, image_size, image_size), dtype=torch.uint8),
        "cls": torch.zeros(batch_size, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]] * batch_size),
        "batch_idx": torch.arange(batch_size, dtype=torch.float32),
    }


def _captured_loss(module: DetectionLitModule, step: str) -> LossResult:
    captured: list[LossResult] = []

    def capture(combined_loss: LossResult, stage: Stage) -> None:
        captured.append(combined_loss)

    module._log_losses = capture  # type: ignore[method-assign]
    getattr(module, step)(_batch(), 0)
    return captured[0]


class TestTrainingStep:
    def test_finite_loss_with_named_components(self) -> None:
        combined = _captured_loss(_module(), "training_step")
        assert torch.isfinite(combined.total)
        assert set(combined.components) == {"boxes/box", "boxes/cls", "boxes/dfl"}

    def test_no_tasks_and_optimizer_over_yolo_parameters(self) -> None:
        module = _module()
        assert module.tasks == []
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]) > 0


class TestValidationStep:
    def test_updates_mean_average_precision(self) -> None:
        module = _module()
        module.eval()  # Lightning puts the module in eval mode for validation
        module._log_losses = lambda combined_loss, stage: None  # type: ignore[method-assign]
        module.validation_step(_batch(), 0)
        assert module._map_accumulator(Stage.VAL).update_count > 0

    def test_epoch_end_logs_and_resets(self) -> None:
        module = _module()
        module.eval()  # Lightning puts the module in eval mode for validation
        module._log_losses = lambda combined_loss, stage: None  # type: ignore[method-assign]
        logged: dict[str, float] = {}
        module.log = lambda key, value, **kwargs: logged.update({key: float(value)})  # type: ignore[method-assign]
        module.validation_step(_batch(), 0)
        module.on_validation_epoch_end()
        assert "boxes/map50/val" in logged
        assert "boxes/map50_95/val" in logged
        assert module._map_accumulator(Stage.VAL).update_count == 0


class TestMetricDirections:
    def test_map_is_higher_is_better(self) -> None:
        directions = _module().metric_directions()
        assert directions["boxes/map50/val"] is True
        assert directions["boxes/map50_95/val"] is True
