"""Every shipped YAML composes and validates: the examples cannot drift from the schema."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import ComponentConfig, ExperimentConfig, WeightedLossConfig, load_config
from src.config.instantiate import resolve_factory
from src.core import Geometry, Registry, Sample, require_tensor
from src.data.build import build_transforms
from src.data.registry import (
    data_module_registry,
    input_encoder_registry,
    preprocessor_registry,
    target_encoder_registry,
)
from src.losses.registry import loss_registry
from src.metrics.registry import metric_registry
from src.models.build import NATIVE
from src.models.registry import backbone_registry, head_registry, model_registry
from src.tasks.registry import task_registry
from src.tracking.registry import tracker_registry
from src.training.registry import learner_registry, optimizer_registry, scheduler_registry
from tests.support.paths import CONFIGS

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


def declared_names(config: ExperimentConfig) -> Iterator[tuple[ComponentConfig, Registry[Any] | None]]:
    """Every name a shipped file writes, beside the registry that has to hold it.

    A transform names no registry: a pixel pipeline is a list of foreign objects reached by import path,
    which `resolve_factory` resolves just the same.
    """
    yield config.model, model_registry
    if config.model.backbone is not None:
        yield config.model.backbone, backbone_registry
    yield config.preprocessing, preprocessor_registry
    yield from ((one, input_encoder_registry) for one in (config.preprocessing.inputs or {}).values())
    yield from ((one, None) for one in config.transforms.values())
    yield config.data, data_module_registry
    yield config.learner, learner_registry
    yield config.optimizer, optimizer_registry
    if config.scheduler is not None:
        yield config.scheduler, scheduler_registry
    if config.tracker is not None:
        yield config.tracker, tracker_registry
    for task in config.tasks.values():
        yield task.kind, task_registry
        if task.target_encoder is not None:
            yield task.target_encoder, target_encoder_registry
        if task.head is not None and task.head.name != NATIVE:
            yield task.head, head_registry
        for one in task.loss if isinstance(task.loss, list) else [task.loss]:
            declared = one.loss if isinstance(one, WeightedLossConfig) else one
            if declared is not None:
                yield declared, loss_registry
        yield from ((one, metric_registry) for one in (task.metrics or {}).values())


def unresolved_names(config: ExperimentConfig) -> list[str]:
    """Every name this configuration writes that nothing implements, with the reason each failed."""
    unresolved = []
    for declared, registry in declared_names(config):
        try:
            resolve_factory(declared, registry)
        except (LookupError, TypeError) as error:
            unresolved.append(f"{declared.spelled}: {error}")
    return unresolved


@pytest.mark.parametrize(
    "override",
    [
        "tracker=none",
        "tracker=csv",
        "tracker=clearml",
        "scheduler=onecycle",
        "scheduler=cosine",
        "scheduler=plateau",
        "scheduler=step",
        "callbacks=default",
        "model=dpt_dinov3",
        "model=unet",
        "trainer=profile",
        "loader=performance",
    ],
)
def test_every_group_option_validates_and_names_something_that_exists(override: str) -> None:
    config = load_config(composed("experiment=examples/classification", override))

    assert unresolved_names(config) == []


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_name_a_shipped_example_writes_resolves_to_an_implementation(example: str) -> None:
    """Validating a file only proves its grammar: a name nothing implements passes and dies at the run."""
    config = load_config(composed(f"experiment=examples/{example}"))

    assert unresolved_names(config) == []
