"""``ExperimentConfig``: the single validated source of truth for one experiment."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config import load_config
from src.core import Objective, Stage, Topology


def make_raw(**overrides: Any) -> dict[str, Any]:
    return {
        "seed": 7,
        "data": {
            "source": "annotations.csv",
            "inputs": {"image": {"column": "path", "loader": "image"}},
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "tasks": {
            "label": {"preset": "classification", "target": "label"},
            "age": {"preset": "regression", "target": "age", "weight": 0.5},
        },
        "model": {"name": "timm", "model_name": "resnet18"},
        "optimizer": {"name": "adamw", "lr": 1.0e-3},
        "loader": {"batch_size": 32, "prefetch_factor": 4},
        "trainer": {"max_epochs": 5, "precision": "16-mixed"},
    } | overrides


def test_a_full_experiment_parses_into_typed_sections() -> None:
    config = load_config(make_raw())

    assert config.seed == 7
    assert config.tasks["label"].topology is Topology.GLOBAL
    assert config.tasks["age"].objective is Objective.CONTINUOUS
    assert config.model.params == {"model_name": "resnet18"}
    assert config.optimizer.params == {"lr": 1.0e-3}


def test_shared_knobs_have_defaults_so_a_minimal_config_loads() -> None:
    config = load_config(make_raw())

    assert config.lr == pytest.approx(1.0e-3)
    assert config.epochs == 10
    assert config.image_size == (224, 224)


def test_the_run_section_defaults_to_train_and_test() -> None:
    config = load_config(make_raw())

    assert config.run.train is True
    assert config.run.test is True
    assert config.run.directory is None


def test_transforms_are_parsed_per_stage() -> None:
    raw = make_raw(transforms={"train": {"_target_": "src.transforms.AlbumentationsTransform", "transforms": []}})

    config = load_config(raw)

    assert config.transforms is not None
    assert str(config.transforms[Stage.TRAIN].target).endswith("AlbumentationsTransform")


def test_a_scheduler_section_is_optional() -> None:
    assert load_config(make_raw()).scheduler is None
    assert load_config(make_raw(scheduler={"name": "cosine", "T_max": 10})).scheduler is not None


def test_forward_sections_keep_unknown_knobs() -> None:
    config = load_config(make_raw())

    assert config.loader.batch_size == 32
    assert (config.loader.model_extra or {})["prefetch_factor"] == 4
    assert (config.trainer.model_extra or {})["precision"] == "16-mixed"


def test_sensible_defaults_cover_optional_sections() -> None:
    raw = make_raw()
    del raw["optimizer"], raw["loader"], raw["trainer"], raw["seed"]

    config = load_config(raw)

    assert config.seed == 42
    assert config.optimizer.name == "adamw"
    assert config.loader.batch_size == 16
    assert config.trainer.max_epochs == 10


def test_an_unknown_root_key_is_rejected_and_named() -> None:
    # A plural slip instead of a misspelling: spell-fixers must not "repair" it.
    raw = make_raw() | {"optimizers": {"name": "sgd"}}

    with pytest.raises(ValidationError, match="optimizers"):
        load_config(raw)


def test_at_least_one_task_is_required() -> None:
    raw = make_raw() | {"tasks": {}}

    with pytest.raises(ValidationError, match="task"):
        load_config(raw)


def test_blank_task_names_are_rejected() -> None:
    raw = make_raw()
    raw["tasks"] = {"  ": {"preset": "classification"}}

    with pytest.raises(ValidationError, match="name"):
        load_config(raw)
