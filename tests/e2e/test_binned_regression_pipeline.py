"""End-to-end: a continuous target learned as a distribution over bins.

The acceptance test for the representation-not-an-axis decision — the task is
declared as ordinary regression, and choosing a binned encoder is what turns the
head, the loss and the read-back into their distributional forms.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from src.assembly import assemble, run
from src.config import load_config
from tests.support.datasets import write_images, write_table

BINS = 12

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
  split: {{train: 0.5, val: 0.25, test: 0.25}}

tasks:
  score:
    preset: regression
    target: score
    target_encoder: {{name: gaussian_bins, bins: 12}}

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


def write_scored(root: Path, rows: int = 24) -> None:
    """Images plus a ``score`` column reading the brightness they were written at.

    The target is learnable from the pixels, so a run that trains at all moves the
    number — which is what the pipeline is being asked to prove.
    """
    records = [
        {"image": image, "score": float(index * 10 % 255) / 255.0}
        for index, image in enumerate(write_images(root, rows))
    ]
    write_table(root, records)


def configure(root: Path) -> object:
    write_scored(root)
    (root / "experiment.yaml").write_text(EXPERIMENT.format(root=root))
    raw = yaml.safe_load((root / "experiment.yaml").read_text())
    return load_config(OmegaConf.to_container(OmegaConf.create(raw), resolve=True))  # type: ignore[arg-type]


@pytest.mark.e2e
def test_a_binned_regression_trains_and_reports_an_ordinary_regression_metric(tmp_path: Path) -> None:
    """Nothing in the task declaration mentions bins: choosing the encoder is the whole change."""
    config = configure(tmp_path)
    experiment = assemble(config)  # type: ignore[arg-type]

    run(experiment, config)  # type: ignore[arg-type]

    assert experiment.trainer.state.finished
    assert experiment.trainer.callback_metrics["test/score/mae"].ndim == 0


@pytest.mark.e2e
def test_both_loss_terms_are_learned_and_logged_apart(tmp_path: Path) -> None:
    """A total that stops falling says less than seeing which of the two terms did."""
    config = configure(tmp_path)
    experiment = assemble(config)  # type: ignore[arg-type]

    run(experiment, config)  # type: ignore[arg-type]

    logged = set(experiment.trainer.callback_metrics)
    assert {"test/score/ce", "test/score/expectation"} <= logged


@pytest.mark.e2e
def test_the_encoder_alone_widens_the_target_to_its_bins(tmp_path: Path) -> None:
    experiment = assemble(configure(tmp_path))  # type: ignore[arg-type]

    batch = next(iter(experiment.data.train_dataloader()))

    assert batch.targets["score"].shape[1] == BINS
