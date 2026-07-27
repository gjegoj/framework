"""Detection regime: our Lightning contour driving an ultralytics YOLO model.

The regime overrides the step methods directly (not ``_shared_step``): ultralytics
batches are their own dict (``img``/``cls``/``bboxes``/``batch_idx``), so the base's
``Batch`` normalization does not apply. Everything else — optimizer configuration,
loss logging, checkpoint pruning, hparams — is inherited from ``BaseLitModule``.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from src.core.entities import Batch, LossResult, StepOutput
from src.core.enums import Stage
from src.models.yolo import compute_detection_loss, decode_predictions, ground_truth_boxes, normalize_batch_images
from src.training.modules.base import BaseLitModule
from src.training.optim.optimizer import OptimizerBuilder
from src.training.optim.scheduler import SchedulerBuilder

_MAP_METRICS = ("map50", "map50_95")
_EVALUATION_STAGES = (Stage.VAL, Stage.TEST)


class DetectionLitModule(BaseLitModule[nn.Module]):
    """YOLO detection training: ultralytics loss on TRAIN, NMS + mAP on VAL/TEST.

    Losses log through the standard contour (``loss/<stage>/total`` plus the
    ``<task>/box|cls|dfl`` components); mAP logs at epoch end as
    ``<task>/map50/<stage>`` and ``<task>/map50_95/<stage>`` — the framework's
    ``task/metric/stage`` grammar, so the progress bar and ClearML parse them as usual.

    Parameters:
        model (nn.Module): YOLO model built by ``build_yolo_model``.
        task_name (str): Task name used in every logged key (e.g. ``boxes``).
        image_size (int): Square image size (converts normalized targets for mAP).
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
        scheduler_builder (SchedulerBuilder | None): Optional LR scheduler builder.
        hparams (dict[str, Any] | None): Config snapshot logged at ``on_fit_start``.
        confidence_threshold (float): Minimum detection confidence kept at evaluation.
        iou_threshold (float): NMS IoU threshold at evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        image_size: int,
        optimizer_builder: OptimizerBuilder,
        scheduler_builder: SchedulerBuilder | None = None,
        hparams: dict[str, Any] | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        super().__init__(
            model=model,
            tasks=[],
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            hparams=hparams,
        )
        self._task_name = task_name
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        # One accumulator per evaluation stage so val and test never mix state.
        self._mean_average_precision = nn.ModuleDict(
            {stage: MeanAveragePrecision(box_format="xyxy") for stage in _EVALUATION_STAGES}
        )

    # ------------------------------------------------------------------ steps

    def _shared_step(self, batch: Any, stage: Stage) -> StepOutput:
        """Not applicable: detection overrides the step methods directly.

        The base's ``_shared_step`` seam assumes the framework ``Batch`` entity;
        ultralytics batches are their own dict, so this regime replaces
        ``training_step``/``validation_step``/``test_step`` instead. Reaching this
        method means a subclass wired the base dispatch back in by mistake.
        """
        raise NotImplementedError("DetectionLitModule overrides the step methods; _shared_step does not apply.")

    def training_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        ultralytics_batch = self._prepare_batch(batch)
        total, components = compute_detection_loss(self.model, ultralytics_batch)
        self._log_detection_loss(total, components, Stage.TRAIN)
        return {"loss": total, "task_views": {}}

    def validation_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._evaluation_step(self._prepare_batch(batch), Stage.VAL)

    def test_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._evaluation_step(self._prepare_batch(batch), Stage.TEST)

    @staticmethod
    def _prepare_batch(batch: Batch | dict[str, Any]) -> dict[str, Any]:
        """Narrow to the ultralytics dict and scale its uint8 images to float ``[0, 1]``."""
        if not isinstance(batch, dict):
            raise TypeError(f"DetectionLitModule expects an ultralytics batch dict, got {type(batch).__name__}.")
        return normalize_batch_images(batch)

    def _evaluation_step(self, batch: dict[str, Any], stage: Stage) -> StepOutput:
        """One forward pass feeds both the loss and the mAP accumulator."""
        with torch.no_grad():
            output = self.model(batch["img"])
        total, components = compute_detection_loss(self.model, batch, predictions=output)
        self._log_detection_loss(total, components, stage)
        predictions = decode_predictions(output, self._confidence_threshold, self._iou_threshold)
        targets = ground_truth_boxes(batch, self._image_size)
        self._map_accumulator(stage).update(predictions, targets)
        return {"loss": total, "task_views": {}}

    def _log_detection_loss(self, total: torch.Tensor, components: dict[str, torch.Tensor], stage: Stage) -> None:
        namespaced = {f"{self._task_name}/{name}": value for name, value in components.items()}
        self._log_losses(LossResult(total=total, components=namespaced), stage)

    # ------------------------------------------------------------ epoch hooks

    def on_validation_epoch_end(self) -> None:
        self._report_mean_average_precision(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._report_mean_average_precision(Stage.TEST)

    def _report_mean_average_precision(self, stage: Stage) -> None:
        accumulator = self._map_accumulator(stage)
        result = accumulator.compute()
        self.log(f"{self._task_name}/map50/{stage}", result["map_50"], prog_bar=True)
        self.log(f"{self._task_name}/map50_95/{stage}", result["map"], prog_bar=True)
        accumulator.reset()

    def _map_accumulator(self, stage: Stage) -> MeanAveragePrecision:
        """The stage's mAP accumulator (ModuleDict lookup erases the element type)."""
        return cast("MeanAveragePrecision", self._mean_average_precision[stage])

    # ---------------------------------------------------------------- utils

    def metric_directions(self) -> dict[str, bool | None]:
        """mAP always improves upward — declared per stage for the progress bar."""
        return {f"{self._task_name}/{metric}/{stage}": True for metric in _MAP_METRICS for stage in _EVALUATION_STAGES}
