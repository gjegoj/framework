"""One run, end to end: a table on disk becomes a trained network and a record of what it learned.

Every phase meets here — encoders, the pixel pipeline a stage declares, the table module, a backbone, a
head sized from both ends, a task's three views, its loss, its metrics, the loop and the tracker. It is
assembled the way a command line assembles it, through the composition root, so a seam that does not
fit fails once with the whole path in view instead of surfacing deep inside a real run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.core import require_tensor
from src.experiment import Experiment, run
from tests.support.declarations import BACKBONE, NORMALIZATION, pixel_pipeline
from tests.support.table import write_table

SIZE = [32, 32]
ENCODER = "learner.model.backbone."


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def declared(table: Path, tmp_path: Path) -> Mapping[str, Any]:
    """A run of the shape the shipped examples have: a task, a schedule, a tracker and a saver."""
    return {
        "seed": 0,
        "epochs": 1,
        "batch_size": 4,
        "data": {
            "name": "table",
            "source": str(table),
            "inputs": {"image": {"column": "image_path"}},
            "split": {
                "train": 0.5,
                "val": 0.25,
                "test": 0.25,
                "rule": {"_target_": "src.data.StratifiedSplit", "by": "species"},
            },
        },
        "preprocessing": {
            "name": "standard",
            "inputs": {"image": {"name": "image", "image_size": SIZE, **NORMALIZATION}},
        },
        "transforms": {stage: pixel_pipeline(SIZE) for stage in ("train", "val", "test")},
        "model": {"name": "composite", "backbone": {"name": "timm", "model_name": BACKBONE, "pretrained": False}},
        "tasks": {"species": {"kind": "classification", "target_column": "species", "classes": {0: "cat", 1: "dog"}}},
        "scheduler": {"name": "cosine", "T_max": 1},
        "tracker": {"name": "csv", "save_dir": str(tmp_path / "run")},
        "callbacks": [
            {
                "name": "checkpoint",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "dirpath": str(tmp_path / "run" / "checkpoints"),
            }
        ],
        "trainer": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "run": {"directory": str(tmp_path / "run"), "project": "tests", "name": "one"},
    }


@pytest.fixture
def experiment(declared: Mapping[str, Any]) -> Experiment:
    return build(load_config(declared))


def test_a_image_on_disk_reaches_a_loss_and_a_gradient_comes_back(experiment: Experiment) -> None:
    learner = experiment.module.learner
    batch = next(iter(experiment.data.train_dataloader()))

    step = learner.step(batch)
    assert step.loss is not None
    step.loss.total.backward()

    assert require_tensor(batch.inputs["image"], name="image").shape == (4, 3, *SIZE)
    assert step.loss.total.ndim == 0 and torch.isfinite(step.loss.total)
    assert set(step.loss.losses) == {"species/cross_entropy"}
    encoder = next(value for name, value in learner.named_parameters() if name.startswith("model.backbone."))
    assert encoder.grad is not None and torch.any(encoder.grad != 0)


def test_the_run_trains_the_network_evaluates_it_and_records_what_it_learned(
    experiment: Experiment, declared: Mapping[str, Any], tmp_path: Path
) -> None:
    """What `uv run main.py` does, minus the command line: the whole run, from a declaration."""
    before = next(value.clone() for name, value in experiment.module.named_parameters() if ENCODER in name)

    run(experiment)

    after = next(value for name, value in experiment.module.named_parameters() if ENCODER in name)
    assert not torch.equal(before, after), "an epoch ran, and the encoder moved"

    recorded = next((tmp_path / "run").rglob("metrics.csv")).read_text()
    assert "val/species/f1/cat" in recorded, "the classes the task declared name the lines a metric drew"
    assert "test/loss" in recorded, "the run evaluated on the split it held out"

    kept = next((tmp_path / "run" / "checkpoints").glob("*.ckpt"))
    assert kept.exists(), "the weights the run settled on are on disk"
