"""End to end: a yes-or-no column trains a binary task on the table pipeline — one logit, BCE, binary metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.build import build, run
from tests.support.configs import experiment_from_yaml
from tests.support.datasets import write_images, write_table

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT = """
data:
  source: {root}/annotations.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25}}
tasks:
  flag:
    kind: binary_classification
    target: flag
    metrics: {{accuracy: {{name: accuracy}}}}
transforms:
  train: &pipeline
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {{_target_: albumentations.Resize, height: 16, width: 16}}
      - {{_target_: albumentations.Normalize}}
      - {{_target_: albumentations.pytorch.ToTensorV2}}
  val: *pipeline
  test: *pipeline
model: {{name: timm, model_name: resnet18, pretrained: false}}
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


def test_a_binary_task_trains_and_tests_on_a_zero_one_column(tmp_path: Path) -> None:
    """The scalar encoder reads the column as it is; the kind's BCE and its binary metrics judge it."""
    write_table(
        tmp_path, [{"image": image, "flag": index % 2} for index, image in enumerate(write_images(tmp_path, 8))]
    )
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path))
    experiment = build(config)

    run(experiment, config)

    logged = set(experiment.trainer.callback_metrics)
    assert "test/flag/bce" in logged
    assert "test/flag/accuracy" in logged
