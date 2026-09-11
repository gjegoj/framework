"""The smallest complete declaration: what a shipped config says, spelled out rather than composed."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.support.declarations import pixel_pipeline
from tests.support.table import write_table

SIZE = [8, 8]
PIPELINE = pixel_pipeline(SIZE)


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("rows"))


@pytest.fixture
def declaration(table: Path, tmp_path: Path) -> Mapping[str, Any]:
    """One classification task over eight pictures, trained for a single step on the processor."""
    return {
        "seed": 0,
        "lr": 1.0e-3,
        "epochs": 1,
        "batch_size": 2,
        "data": {
            "name": "table",
            "source": str(table),
            "inputs": {"image": {"column": "image_path"}},
            "split": {"train": 0.5, "val": 0.25, "test": 0.25},
        },
        "preprocessing": {"name": "standard", "inputs": {"image": {"name": "image", "image_size": SIZE}}},
        "transforms": {"train": PIPELINE, "val": PIPELINE, "test": PIPELINE},
        "model": {"name": "composite", "backbone": {"name": "timm", "model_name": "resnet18", "pretrained": False}},
        "tasks": {"species": {"kind": "classification", "target": "species", "classes": {0: "cat", 1: "dog"}}},
        "trainer": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "run": {"directory": str(tmp_path / "run"), "project": "tests", "name": "one"},
    }
