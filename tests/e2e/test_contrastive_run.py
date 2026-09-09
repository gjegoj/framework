"""End to end: two streams aligned by comparison — two inputs, two encoders, InfoNCE over the stacked embeddings, no target column."""

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
    view: {{column: view, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25}}
tasks:
  pair: {{kind: contrastive}}
transforms:
  train: &pipeline
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {{_target_: albumentations.Resize, height: 16, width: 16}}
      - {{_target_: albumentations.Normalize}}
      - {{_target_: albumentations.pytorch.ToTensorV2}}
  val: *pipeline
  test: *pipeline
model:
  name: multi
  embedding_dim: 8
  encoders:
    image: {{_target_: src.models.TimmBackbone, model_name: resnet18, pretrained: false}}
    view: {{_target_: src.models.TimmBackbone, model_name: resnet18, pretrained: false, input_name: view}}
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


def test_a_contrastive_task_trains_and_tests_on_two_image_columns(tmp_path: Path) -> None:
    """Nested encoders build by ``_target_``; the kind needs no target, and the batch's own structure supervises."""
    images = write_images(tmp_path, 8)
    write_table(tmp_path, [{"image": image, "view": image} for image in images])
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path))
    experiment = build(config)

    run(experiment, config)

    assert "test/pair/infonce" in experiment.trainer.callback_metrics
