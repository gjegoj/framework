"""Every experiment this project ships composes, validates, and is one the composition root accepts.

A config file nobody loads rots in silence, and two that shipped before this test did:
one wrote ``metrics: {accuracy: {}}``, an entry naming no metric; another kept a
transforms section the model it named could not serve. Both sat in the repository as the
framework's own worked examples.

Neither is caught by composing alone: validation is what catches them, and it needs no
dataset — so every shipped example is composed and validated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.build import build
from src.config import load_config
from tests.support.datasets import write_pet_like
from tests.support.fakes import clearml_stub

CONFIGS = Path(__file__).parents[3] / "configs"
EXPERIMENTS = sorted(path.stem for path in (CONFIGS / "experiment" / "examples").glob("*.yaml"))

BASE = "pet"
"""The shared base, which declares no task of its own and is not a run."""


def test_the_examples_folder_is_not_empty() -> None:
    """A glob that silently matched nothing would make every test below vacuous."""
    assert len(EXPERIMENTS) > 1


@pytest.mark.parametrize("experiment", [name for name in EXPERIMENTS if name != BASE])
def test_a_shipped_experiment_composes_into_a_valid_config(experiment: str) -> None:
    """What is offered as a worked example has to be one a user can actually run."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(
            config_name="config",
            overrides=[f"experiment=examples/{experiment}", "run.directory=."],
        )

    config = load_config(cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True)))

    assert config.tasks, "an example with no task trains nothing"


@pytest.mark.parametrize("experiment", [name for name in EXPERIMENTS if name != BASE])
def test_a_shipped_experiment_builds_over_a_pet_shaped_dataset(
    experiment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validating proves the YAML; building proves the model, losses, metrics and callbacks it names exist together.

    Before this, ``make test-run`` by hand over the downloaded dataset was the only thing
    that built an example. Only the source, the weights, the size and the run's home change.
    """
    clearml_stub(monkeypatch)  # `all_callbacks` tracks with clearml
    table = write_pet_like(tmp_path)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(
            config_name="config",
            overrides=[
                f"experiment=examples/{experiment}",
                f"data.source={table}",
                f"run.directory={tmp_path / 'run'}",
                "model.pretrained=false",
                "image_size=[32,32]",
                "epochs=1",
            ],
        )
    config = load_config(cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True)))

    built = build(config)

    assert {task.name for task in built.module.tasks} == set(config.tasks)


def test_the_shared_base_declares_no_task_of_its_own() -> None:
    """It is inherited, never run — and a base carrying a task would give every example one."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", overrides=[f"experiment=examples/{BASE}"])

    assert not composed.tasks
