"""The acceptance test for assembly: a YAML file on disk trains for one epoch.

This is the first time the framework runs the way users will run it — the CLI
does nothing this test does not do, beyond letting Hydra compose the mapping.
"""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from src.assembly import Experiment, assemble, run
from src.config import load_config
from tests.support.datasets import write_dataset

ONE_SOURCE = """data:
  source: {root}/annotations.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25}}
"""

CACHED_SOURCE = """data:
  source: {root}/annotations.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25}}
  cache: {{name: ram, max_gib: 0.25, workers: 2}}
"""

TWO_SOURCES = """data:
  source:
    - {root}/clean.csv
    - path: {root}/noisy.csv
      transforms:
        train:
          _target_: src.transforms.AlbumentationsTransform
          transforms:
            - {{_target_: albumentations.Resize, height: 16, width: 16}}
            - {{_target_: albumentations.HorizontalFlip, p: 1.0}}
            - {{_target_: albumentations.Normalize}}
            - {{_target_: albumentations.pytorch.ToTensorV2}}
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
  split: {{train: 0.5, val: 0.25, test: 0.25, stratify_by: label}}
"""

PER_STAGE_SOURCES = """data:
  source:
    train: {root}/train.csv
    val: {root}/val.csv
    test: {root}/test.csv
  inputs:
    image: {{column: image, loader: {{name: image, root: {root}}}}}
"""

EXPERIMENT = """
seed: 7
lr: 1.0e-3
epochs: 1
batch_size: 2
image_size: [16, 16]

{data}
tasks:
  label:
    preset: classification
    target: label
    metrics: {{accuracy: {{name: accuracy}}}}

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
optimizer: {{name: sgd, lr: "${{lr}}"}}
scheduler: {{name: onecycle, interval: step, max_lr: "${{lr}}"}}
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


def written(root: Path, rows: int = 8) -> pd.DataFrame:
    """The dataset, and its table read back — what a pre-split test divides by hand."""
    return pd.read_csv(write_dataset(root, rows))


def run_experiment(root: Path, data: str) -> Experiment:
    (root / "experiment.yaml").write_text(EXPERIMENT.format(root=root, data=data.format(root=root)))

    raw = yaml.safe_load((root / "experiment.yaml").read_text())
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    config = load_config(resolved)  # type: ignore[arg-type]

    experiment = assemble(config)
    run(experiment, config)
    return experiment


@pytest.mark.e2e
def test_a_yaml_experiment_trains_and_tests(tmp_path: Path) -> None:
    written(tmp_path)
    (tmp_path / "experiment.yaml").write_text(EXPERIMENT.format(root=tmp_path, data=ONE_SOURCE.format(root=tmp_path)))

    raw = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    config = load_config(resolved)  # type: ignore[arg-type]

    experiment = assemble(config)
    run(experiment, config)

    assert experiment.trainer.state.finished
    assert "test/label/accuracy" in experiment.trainer.callback_metrics
    assert experiment.trainer.lr_scheduler_configs != []


@pytest.mark.e2e
def test_a_yaml_experiment_runs_on_sources_that_are_already_divided(tmp_path: Path) -> None:
    """A partition decided upstream reaches the run intact — no fractions, no re-cutting."""
    frame = written(tmp_path)
    stages = {"train": frame.iloc[:4], "val": frame.iloc[4:6], "test": frame.iloc[6:]}
    for stage, rows in stages.items():
        rows.to_csv(tmp_path / f"{stage}.csv", index=False)

    experiment = run_experiment(tmp_path, PER_STAGE_SOURCES)

    assert experiment.trainer.state.finished
    assert "test/label/accuracy" in experiment.trainer.callback_metrics
    datasets = [
        experiment.data.train_dataloader().dataset,
        experiment.data.val_dataloader().dataset,
        experiment.data.test_dataloader().dataset,
    ]
    assert [len(cast("Sized", dataset)) for dataset in datasets] == [4, 2, 2]


@pytest.mark.e2e
def test_two_sources_each_with_its_own_transforms_train_together(tmp_path: Path) -> None:
    """Datasets that need different handling are combined without leaving the one pipeline."""
    frame = written(tmp_path, rows=16)
    frame.iloc[:8].to_csv(tmp_path / "clean.csv", index=False)
    frame.iloc[8:].to_csv(tmp_path / "noisy.csv", index=False)

    experiment = run_experiment(tmp_path, TWO_SOURCES)

    assert experiment.trainer.state.finished
    assert "test/label/accuracy" in experiment.trainer.callback_metrics
    datasets = [
        experiment.data.train_dataloader().dataset,
        experiment.data.val_dataloader().dataset,
        experiment.data.test_dataloader().dataset,
    ]
    assert sum(len(cast("Sized", dataset)) for dataset in datasets) == 16


@pytest.mark.e2e
def test_a_cached_experiment_trains_and_tests(tmp_path: Path) -> None:
    """A cache is an optimisation: the run has to behave exactly as it would without one."""
    written(tmp_path)

    experiment = run_experiment(tmp_path, CACHED_SOURCE)

    assert experiment.trainer.state.finished
    assert "test/label/accuracy" in experiment.trainer.callback_metrics
