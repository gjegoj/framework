"""End to end: any number of classes per pixel is declared, and refused at build until an encoder exists.

No shipped encoder produces a multi-hot ``[C, H, W]`` target — ``mask`` yields an index map, one
class per pixel — so the kind says so up front (``unavailable``) instead of building a model
that dies at its first step. The day the encoder lands, this test turns into a training run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.build import build
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
    kind: multilabel_segmentation
    target: mask
    target_encoder: {{name: mask, root: {root}}}
    classes: {{0: background, 1: pet, 2: boundary}}
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


def test_a_multilabel_segmentation_task_is_refused_at_build_naming_what_is_missing(tmp_path: Path) -> None:
    write_dataset(tmp_path, masks=True, mask_classes=3)
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path))

    with pytest.raises(ValueError, match=r"Task 'mask' is 'MultilabelSegmentation'.*multi-hot"):
        build(config)
