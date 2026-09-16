"""Declarations built from a few overrides, so a test names only what it is about.

Two kinds live here, and the shipped ones are declarations too: a run written out by hand, and one
composed from `configs/` as a reader composes it. Both are read by more than one test, and a second
copy of either is a second statement of what a run looks like.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import ComponentConfig, PreprocessingConfig, TaskConfig
from tests.support.paths import CONFIGS

EXAMPLES = sorted(path.stem for path in (CONFIGS / "experiment" / "examples").glob("*.yaml") if path.stem != "pet")
"""Every shipped example but the preset the others inherit, which is not a run on its own."""

CLASSES = {0: "cat", 1: "dog"}
SIZE = [8, 8]

BACKBONE = "test_resnet"
"""The network these declarations are built over: timm's own smallest, kept for exactly this.

Everything assembled from `smallest_run` is about something other than the network — a callback, a
loop, a record, a page — and pays for one on every run all the same. Measured 2026-09-16: `resnet18`
holds 11.18 M parameters against this one's 0.37 M, which is about 190 MB a run once gradients and two
optimizer states stand beside the weights, and the suite reached the memory of the machine it runs on
and was killed there. The tests that are *about* timm — a backbone, an adapter, a foreign file — name
a real architecture, because for those the architecture is the subject."""
NORMALIZATION = {"mean": [0.5] * 3, "std": [0.5] * 3}
"""What these fixtures scale an image by — one source, read by the encoder that declares it and by the
chain that applies it. The shipped groups interpolate the same two numbers from one place for the same
reason: nothing checks that the halves agree, so writing them twice is how they come to differ."""


def composed(*overrides: str, directory: str = "runs/test") -> Mapping[str, Any]:
    """The shipped tree composed as a reader composes it, resolved into a plain declaration.

    ``${hydra:run.dir}`` resolves only inside a Hydra job, so the run directory is given by hand — and
    given by the caller, because a test that builds what it composed writes files under it.
    """
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        composed_config = compose(config_name="config", overrides=[*overrides, f"run.directory={directory}"])
        raw = OmegaConf.to_container(composed_config, resolve=True)
    assert isinstance(raw, dict)
    return cast("Mapping[str, Any]", raw)


def pixel_pipeline(size: list[int]) -> dict[str, Any]:
    """What `configs/transforms/default.yaml` declares, written out rather than composed by Hydra."""
    return {
        "_target_": "src.transforms.AlbumentationsTransform",
        "transforms": [
            {"_target_": "albumentations.Resize", "height": size[0], "width": size[1]},
            {"_target_": "albumentations.Normalize", **NORMALIZATION},
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
        {"name": "standard", "inputs": {"image": {"name": "image", "image_size": [4, 4], **NORMALIZATION}}, **declared}
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
        "preprocessing": {
            "name": "standard",
            "inputs": {"image": {"name": "image", "image_size": SIZE, **NORMALIZATION}},
        },
        "transforms": {stage: pixel_pipeline(SIZE) for stage in ("train", "val", "test")},
        "model": {"name": "composite", "backbone": {"name": "timm", "model_name": BACKBONE, "pretrained": False}},
        "tasks": {"species": {"kind": "classification", "target_column": "species", "classes": CLASSES}},
        "trainer": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "run": {"directory": str(directory), "project": "tests", "name": "one"},
    }
