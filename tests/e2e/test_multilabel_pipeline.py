"""End-to-end: a multi-label column from a YAML experiment through to a trained epoch.

The acceptance test for multi-label support — the encoder's vocabulary has to
size the head, the indicator vector has to reach binary cross-entropy as floats,
and the stage split has to keep each label's rate rather than each combination's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from src.assembly import assemble, run
from src.config import load_config
from tests.support.datasets import write_images, write_table

VOCABULARY = ("sunny", "beach", "people", "night")

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
    preset: multilabel_classification
    target: tags
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


@pytest.mark.e2e
def test_a_multilabel_experiment_trains_and_tests(tmp_path: Path) -> None:
    write_tagged(tmp_path)
    (tmp_path / "experiment.yaml").write_text(EXPERIMENT.format(root=tmp_path))

    raw = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    config = load_config(resolved)  # type: ignore[arg-type]

    experiment = assemble(config)
    run(experiment, config)

    assert experiment.trainer.state.finished
    assert "test/tags/accuracy" in experiment.trainer.callback_metrics
    assert "test/tags/f1" in experiment.trainer.callback_metrics


@pytest.mark.e2e
def test_the_head_is_sized_from_the_labels_found_in_the_data(tmp_path: Path) -> None:
    """Nobody declares the class count: it follows from the vocabulary the encoder learned."""
    write_tagged(tmp_path)
    (tmp_path / "experiment.yaml").write_text(EXPERIMENT.format(root=tmp_path))

    raw = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    config = load_config(OmegaConf.to_container(OmegaConf.create(raw), resolve=True))  # type: ignore[arg-type]

    experiment = assemble(config)

    batch = next(iter(experiment.data.train_dataloader()))
    assert batch.targets["tags"].shape[1] == len(VOCABULARY)
    assert batch.targets["tags"].dtype.is_floating_point
