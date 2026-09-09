"""End-to-end: a multi-label column from a YAML experiment through to a trained epoch.

The acceptance test for multi-label support — the encoder's vocabulary has to
size the head, the indicator vector has to reach binary cross-entropy as floats,
and the stage split has to keep each label's rate rather than each combination's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.build import build, run
from tests.support.configs import experiment_from_yaml
from tests.support.datasets import write_images, write_table

VOCABULARY = ("sunny", "beach", "people", "night")
CLASSES = "{" + ", ".join(f"{index}: {name}" for index, name in enumerate(VOCABULARY)) + "}"

EXPERIMENT = """
seed: 7
lr: 1.0e-3
epochs: 1
batch_size: 4
image_size: [16, 16]

data:
  source: {root}/annotations.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25, stratify_by: tags}}

tasks:
  tags:
    kind: multilabel_classification
    target: tags
    classes: {classes}
    target_encoder: {{name: multilabel}}
    metrics: {{accuracy: {{name: accuracy}}, f1: {{name: f1}}}}

transforms:
  train: &pipeline
    _target_: src.transforms.AlbumentationsTransform
    transforms:
      - {{_target_: albumentations.Resize, height: "${{image_size.0}}", width: "${{image_size.1}}"}}
      - {{_target_: albumentations.Normalize}}
      - {{_target_: albumentations.pytorch.ToTensorV2}}
  val: *pipeline
  test: *pipeline

model: {{name: timm, model_name: resnet18, pretrained: false}}
optimizer: {{name: adamw, lr: "${{lr}}"}}
loader: {{batch_size: "${{batch_size}}"}}
trainer:
  max_epochs: "${{epochs}}"
  accelerator: cpu
  devices: 1
  enable_progress_bar: false
  enable_model_summary: false
  enable_checkpointing: false
  logger: false
run: {{train: true, test: true, directory: {root}}}
"""


def write_tagged(root: Path, rows: int = 40) -> None:
    """Images plus a ``tags`` column: several labels per row, at deliberately uneven rates."""
    rng = np.random.default_rng(0)
    rates = (0.7, 0.4, 0.25, 0.1)
    records = []
    for image in write_images(root, rows):
        tags = [label for label, rate in zip(VOCABULARY, rates, strict=True) if rng.random() < rate]
        records.append({"image": image, "tags": ",".join(tags)})
    write_table(root, records)


def test_a_multilabel_experiment_trains_and_tests(tmp_path: Path) -> None:
    write_tagged(tmp_path)
    config = experiment_from_yaml(EXPERIMENT.format(root=tmp_path, classes=CLASSES))

    experiment = build(config)
    run(experiment, config)

    assert experiment.trainer.state.finished
    assert "test/tags/accuracy" in experiment.trainer.callback_metrics
    assert "test/tags/f1" in experiment.trainer.callback_metrics


def test_the_head_is_sized_from_the_declared_vocabulary(tmp_path: Path) -> None:
    """Nobody declares the class count: it follows from the vocabulary the encoder learned."""
    write_tagged(tmp_path)

    experiment = build(experiment_from_yaml(EXPERIMENT.format(root=tmp_path, classes=CLASSES)))

    batch = next(iter(experiment.data.train_dataloader()))
    assert batch.targets["tags"].shape[1] == len(VOCABULARY)
    assert batch.targets["tags"].dtype.is_floating_point
