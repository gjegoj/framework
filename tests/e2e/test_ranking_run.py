"""End to end: N views of one item through a shared encoder — the multi-view transform, the multiview backbone, InfoNCE."""

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
  item: {{kind: ranking}}
transforms:
  train: &views
    _target_: src.transforms.MultiViewTransform
    views: 2
    base:
      _target_: src.transforms.AlbumentationsTransform
      transforms:
        - {{_target_: albumentations.Resize, height: 16, width: 16}}
        - {{_target_: albumentations.HorizontalFlip, p: 0.5}}
        - {{_target_: albumentations.Normalize}}
        - {{_target_: albumentations.pytorch.ToTensorV2}}
  val: *views
  test: *views
model:
  name: multiview
  embedding_dim: 8
  inner: {{_target_: src.models.TimmBackbone, model_name: resnet18, pretrained: false}}
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


def test_a_ranking_task_trains_and_tests_on_stacked_views(tmp_path: Path) -> None:
    """The transform stacks the views, the backbone encodes them with one inner network, the loss compares them."""
    write_dataset(tmp_path)
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path))
    experiment = build(config)

    run(experiment, config)

    assert "test/item/infonce" in experiment.trainer.callback_metrics
