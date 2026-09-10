"""Every shipped YAML composes and validates: the examples cannot drift from the schema."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import ExperimentConfig, load_config
from src.core import Geometry, Sample, require_tensor
from src.data.build import build_transforms

CONFIGS = Path(__file__).parents[2] / "configs"
EXAMPLES = sorted(path.stem for path in (CONFIGS / "experiment" / "examples").glob("*.yaml") if path.stem != "pet")


def composed(*overrides: str) -> Mapping[str, Any]:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        # ``${hydra:run.dir}`` resolves only inside a Hydra job; a run directory is given by hand here.
        composed_config = compose(config_name="config", overrides=[*overrides, "run.directory=runs/test"])
        raw = OmegaConf.to_container(composed_config, resolve=True)
    assert isinstance(raw, dict)
    return cast(Mapping[str, Any], raw)


@pytest.mark.parametrize("example", EXAMPLES)
def test_a_shipped_example_composes_into_a_valid_experiment(example: str) -> None:
    config = load_config(composed(f"experiment=examples/{example}"))

    assert isinstance(config, ExperimentConfig) and config.tasks


@pytest.mark.parametrize(
    "override",
    [
        "tracker=clearml",
        "tracker=none",
        "scheduler=onecycle",
        "callbacks=default",
        "export=all",
        "adapters=lora",
        "distillation=kl",
    ],
)
def test_every_group_option_validates_on_the_classification_example(override: str) -> None:
    load_config(composed("experiment=examples/classification", override))


@pytest.mark.parametrize("group", ["default", "augmented"])
def test_a_shipped_transforms_group_prepares_a_picture_for_the_declared_size(group: str) -> None:
    """The pixel pipeline lives in YAML, so a wrong import path or a dropped tensor step must fail here."""
    config = load_config(composed("experiment=examples/classification", f"transforms={group}"))
    geometries = {"inputs": {"image": Geometry.IMAGE}, "targets": {"mask": Geometry.MASK}, "auxiliary_inputs": {}}

    built = build_transforms(config.transforms, geometries)

    sample = Sample(inputs={"image": np.zeros((6, 8, 3), np.uint8)}, targets={"mask": np.zeros((6, 8), np.int64)})
    for stage, prepare in built.items():
        prepared = prepare(sample)
        image = require_tensor(prepared.inputs["image"], name=stage)
        assert image.shape == (3, 224, 224) and image.dtype.is_floating_point
        assert require_tensor(prepared.targets["mask"], name="mask").shape == (224, 224)
