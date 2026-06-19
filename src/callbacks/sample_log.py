"""Callback that renders predictions vs ground truth to interactive HTML.

The producer of the visualization pipeline: it gates by epoch/batch index,
denormalizes the input image, builds one ``SampleView`` per batch element by
delegating to per-task ``Annotator`` strategies, hands them to an injected
``Renderer`` (HTML by default), and ships the HTML through ``PlotLogger.log_html``.
Knows nothing about ClearML or Plotly — both live behind boundaries.

Replaces the MVP ``ImageGridLogCallback``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor

from src.core.constants import IMAGENET_MEAN, IMAGENET_STD
from src.core.entities import Batch, is_training_step_output
from src.core.enums import Stage
from src.core.ports import PlotLogger
from src.training.modules import LitModule
from src.visualization.pipeline import build_sample_views
from src.visualization.renderer import HtmlRenderer, Renderer

log = logging.getLogger(__name__)


class SampleLogCallback(L.Callback):
    """Log a grid of input images with per-task gt/pred labels as interactive HTML.

    Parameters:
        num_images (int): How many batch samples to include in the grid.
        every_n_epochs (int): Log every N epochs (0, N, 2N, ...).
        batch_index (int): Which batch index within the epoch to log.
        mean (list[float] | None): Denormalization mean (default ImageNet).
        std (list[float] | None): Denormalization std (default ImageNet).
        title_prefix (str): Prefix for the logged grid title.
        renderer (Renderer | None): IR → HTML renderer (default ``HtmlRenderer``).
    """

    def __init__(
        self,
        num_images: int = 8,
        every_n_epochs: int = 5,
        batch_index: int = 0,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        title_prefix: str = "samples",
        renderer: Renderer | None = None,
    ) -> None:
        super().__init__()
        if num_images <= 0:
            raise ValueError(f"SampleLogCallback: num_images must be positive, got {num_images}.")
        if every_n_epochs <= 0:
            raise ValueError(f"SampleLogCallback: every_n_epochs must be positive, got {every_n_epochs}.")
        if batch_index < 0:
            raise ValueError(f"SampleLogCallback: batch_index must be >= 0, got {batch_index}.")
        self._num_images = num_images
        self._every_n_epochs = every_n_epochs
        self._batch_index = batch_index
        self._mean = torch.tensor(mean, dtype=torch.float32)
        self._std = torch.tensor(std, dtype=torch.float32)
        self._title_prefix = title_prefix
        self._renderer: Renderer = renderer if renderer is not None else HtmlRenderer()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: object,
        batch_idx: int,
    ) -> None:
        self._maybe_log(trainer, pl_module, outputs, batch, Stage.TRAIN, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_log(trainer, pl_module, outputs, batch, Stage.VAL, batch_idx)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_log(trainer, pl_module, outputs, batch, Stage.TEST, batch_idx)

    def _should_log(self, epoch: int, batch_idx: int) -> bool:
        return epoch % self._every_n_epochs == 0 and batch_idx == self._batch_index

    def _maybe_log(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: object,
        stage: Stage,
        batch_idx: int,
    ) -> None:
        if not self._should_log(trainer.current_epoch, batch_idx):
            return
        if not is_training_step_output(outputs):
            return
        if not isinstance(trainer.logger, PlotLogger):
            return
        if not isinstance(pl_module, LitModule):
            return

        batch_obj = batch if isinstance(batch, Batch) else Batch(**batch) if isinstance(batch, dict) else None
        if batch_obj is None or not batch_obj.inputs:
            return

        first_tensor = next(iter(batch_obj.inputs.values()))
        if not isinstance(first_tensor, Tensor) or first_tensor.ndim != 4:
            return

        images = _denormalize_to_uint8(first_tensor, self._num_images, self._mean, self._std).numpy()
        samples = build_sample_views(images, pl_module.tasks, outputs["task_views"])
        title = f"{self._title_prefix}/{stage}"
        html = self._renderer.render(samples, title=title)
        trainer.logger.log_html(title=title, html=html, iteration=trainer.current_epoch)
        log.debug("Logged sample grid for %s epoch %s.", stage, trainer.current_epoch)


def _denormalize_to_uint8(tensor: Tensor, num_images: int, mean: Tensor, std: Tensor) -> Tensor:
    """Convert a normalized ``[B, C, H, W]`` float tensor to ``[N, H, W, C]`` uint8 RGB."""
    x = tensor[:num_images].detach().cpu().float()
    channels = x.shape[1]
    mean_view = mean[:channels].view(1, channels, 1, 1)
    std_view = std[:channels].view(1, channels, 1, 1)
    x = (x * std_view + mean_view).clamp(0.0, 1.0).mul(255).byte()
    if channels == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.permute(0, 2, 3, 1)
