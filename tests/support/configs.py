"""The two smallest experiments a test builds on: one that validates, one that runs.

``paper_config`` touches no disk and is what a wiring test wants — which callback got
built, which scheduler, what a vendor family refuses. ``disk_config`` names files
``write_dataset`` wrote and is what an assembling or fitting test wants.

**A named section replaces the default whole.** That one rule covers both "use a
different scheduler" and "use a different data section"; to *extend* a default rather
than replace it, compose with the pieces below::

    paper_config(callbacks=[{"name": "lr_monitor"}])          # a section the base omits
    paper_config(data=DATA | {"cache": {"name": "ram"}})      # one key added to a default
    paper_config(tasks={"person": TASK | {"target": "person_id"}})

Nothing here deep-merges. A test that changes a section shows the section it changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from src.config import load_config
from tests.support.datasets import IMAGE_SIDE, TABLE_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import ExperimentConfig

INPUTS: Final[dict[str, Any]] = {"image": {"column": "image"}}
"""One image input, read from the column ``write_dataset`` writes."""

SPLIT: Final[dict[str, Any]] = {"train": 0.5, "val": 0.25, "test": 0.25}
"""Fractions that divide eight rows into four, two and two."""

DATA: Final[dict[str, Any]] = {"source": "a.csv", "inputs": INPUTS, "split": SPLIT}
"""The smallest data section that validates. Its source is never read."""

TASK: Final[dict[str, Any]] = {"preset": "classification", "target": "label"}
"""One classification task over the column ``write_dataset`` writes."""

TASKS: Final[dict[str, Any]] = {"label": TASK}
"""That task under the name its metrics and losses log by."""

MODEL: Final[dict[str, Any]] = {"name": "timm", "model_name": "resnet18", "pretrained": False}
"""A small timm backbone, never downloaded."""


def resizing_transforms(side: int = IMAGE_SIDE) -> dict[str, Any]:
    """A per-stage pipeline ending where every pipeline must: normalised, and a tensor.

    The size is an argument rather than a constant because it is not free: a run whose
    model expects 32 and whose pipeline resizes to 16 fails inside torch, several layers
    from the declaration that caused it. ``disk_config`` therefore derives this from the
    ``image_size`` the test declared, so the two cannot disagree.
    """
    return {
        stage: {
            "_target_": "src.transforms.AlbumentationsTransform",
            "transforms": [
                {"_target_": "albumentations.Resize", "height": side, "width": side},
                {"_target_": "albumentations.Normalize"},
                {"_target_": "albumentations.pytorch.ToTensorV2"},
            ],
        }
        for stage in ("train", "val", "test")
    }


def disk_data(root: Path, **keys: Any) -> dict[str, Any]:
    """The data section naming files under ``root`` — the one section a root changes.

    Exported so a test that varies the *source* keeps the inputs and the split it is not
    about: ``disk_data(root, source="a.parquet")``.
    """
    return {
        "source": str(root / TABLE_NAME),
        "inputs": {"image": {"column": "image", "loader": {"name": "image", "root": str(root)}}},
        "split": SPLIT,
    } | keys


def paper_config(**sections: Any) -> ExperimentConfig:
    """The smallest experiment that validates — for tests about wiring, not about data.

    Its ``data.source`` is a name nobody opens, which is the point: a test asking which
    callback assembly built should not need a dataset to find out.
    """
    return load_config({"data": DATA, "tasks": TASKS, "model": MODEL} | sections)


def disk_config(root: Path, **sections: Any) -> ExperimentConfig:
    """The smallest experiment that assembles and fits, over files ``write_dataset`` wrote.

    Parameters:
        root (Path): The dataset root — where the table sits and what paths resolve against.
        **sections (Any): Sections replacing the defaults whole.

    A config written as a mapping has no interpolation, so the line the shipped groups
    carry is applied here instead: ``default_root_dir: ${run.directory}``. Without it a
    fitting test writes Lightning's own ``lightning_logs/`` into the working directory —
    the repository root — because that is what an unset root falls back to.
    """
    height, _ = sections.get("image_size") or (IMAGE_SIDE, IMAGE_SIDE)
    declared = {
        "data": disk_data(root),
        "tasks": TASKS,
        "model": MODEL,
        "loader": {"batch_size": 2},
        "trainer": {"max_epochs": 1},
        "transforms": resizing_transforms(height),
    } | sections
    return load_config(declared | {"trainer": _rooted(declared)})


def _rooted(declared: dict[str, Any]) -> dict[str, Any]:
    """The trainer section with its root filled from the run's directory, as YAML would."""
    trainer = dict(declared.get("trainer") or {"max_epochs": 1})
    directory = (declared.get("run") or {}).get("directory")
    if directory is not None and "default_root_dir" not in trainer:
        trainer["default_root_dir"] = directory
    return trainer
