"""Validated declarations built from a few overrides, so a test names only what it is about."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import ComponentConfig, PreprocessingConfig, TaskConfig

CLASSES = {0: "cat", 1: "dog"}
SIZE = [8, 8]


def pixel_pipeline(size: list[int]) -> dict[str, Any]:
    """What `configs/transforms/default.yaml` declares, written out rather than composed by Hydra."""
    return {
        "_target_": "src.transforms.AlbumentationsTransform",
        "transforms": [
            {"_target_": "albumentations.Resize", "height": size[0], "width": size[1]},
            {"_target_": "albumentations.Normalize", "mean": [0.5] * 3, "std": [0.5] * 3},
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ],
    }


def task_config(**declared: Any) -> TaskConfig:
    """A classification task on the ``species`` column unless a key says otherwise."""
    return TaskConfig.model_validate({"kind": "classification", "target_column": "species", **declared})


def component(declared: Any) -> ComponentConfig:
    return ComponentConfig.model_validate(declared)


def preprocessing_config(**declared: Any) -> PreprocessingConfig:
    """The standard image preprocessor at 4x4, plus whatever a test adds."""
    return PreprocessingConfig.model_validate(
        {"name": "standard", "inputs": {"image": {"name": "image", "image_size": [4, 4]}}, **declared}
    )


def smallest_run(table: Path, directory: Path) -> dict[str, Any]:
    """One classification task over eight images: what a shipped config says, spelled out.

    Complete rather than minimal — this is what `build` is handed — so a test that needs a real run to
    watch (a callback, a loop) writes only the line it is about and gets the rest of a run around it.
    """
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
        "transforms": {stage: pixel_pipeline(SIZE) for stage in ("train", "val", "test")},
        "model": {"name": "composite", "backbone": {"name": "timm", "model_name": "resnet18", "pretrained": False}},
        "tasks": {"species": {"kind": "classification", "target_column": "species", "classes": CLASSES}},
        "trainer": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "run": {"directory": str(directory), "project": "tests", "name": "one"},
    }
