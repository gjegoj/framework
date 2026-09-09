"""End to end: a two-class mask trains a binary dense task — one channel, BCE per pixel, iou."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.build import build, run
from tests.support.configs import experiment_from_yaml
from tests.support.datasets import write_dataset

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT = """
data:
  source: {root}/annotations.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25}}
tasks:
  mask:
    kind: binary_segmentation
    target: mask
    target_encoder: {{name: mask, root: {root}}}
    classes: {{0: background, 1: pet}}
transforms:
  train: &pipeline
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {{_target_: albumentations.Resize, height: 32, width: 32}}
      - {{_target_: albumentations.Normalize}}
      - {{_target_: albumentations.pytorch.ToTensorV2}}
  val: *pipeline
  test: *pipeline
model: {{name: smp, arch: unet, encoder_name: resnet18, pretrained: false}}
optimizer: {{name: sgd, lr: 1.0e-3}}
loader: {{batch_size: 2}}
trainer:
  max_epochs: 1
  accelerator: cpu
  devices: 1
  enable_progress_bar: false
  enable_model_summary: false
  enable_checkpointing: false
  logger: false
run: {{train: true, test: true, directory: {root}/run}}
"""


def test_a_binary_segmentation_trains_and_tests_from_two_class_masks(tmp_path: Path) -> None:
    """The mask encoder reports two classes, the kind projects one channel, and iou reads it as foreground or not."""
    write_dataset(tmp_path, masks=True, mask_classes=2)
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path))
    experiment = build(config)

    run(experiment, config)

    logged = set(experiment.trainer.callback_metrics)
    assert "test/mask/bce" in logged
    assert any(key.startswith("test/mask/iou") for key in logged)
