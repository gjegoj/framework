"""Every experiment this project ships composes, validates, and is one assembly accepts.

A config file nobody loads rots in silence, and two that shipped before this test did:
one wrote ``metrics: {accuracy: {}}``, an entry naming no metric; the detection one kept
the default transforms, which a vendor family refuses because it augments through its
own. Both sat in the repository as the framework's own worked examples.

Neither is caught by composing alone — the first by validation, the second only by the
refusal ``assemble`` runs before it reads anything. So both gates are applied here, and
neither needs a dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.assembly.vendor import refuse_what_a_vendor_cannot_serve
from src.config import load_config

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

    # The first thing `assemble` does, and the only one that needs no data on disk.
    refuse_what_a_vendor_cannot_serve(config)

    assert config.tasks, "an example with no task trains nothing"


def test_the_shared_base_declares_no_task_of_its_own() -> None:
    """It is inherited, never run — and a base carrying a task would give every example one."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", overrides=[f"experiment=examples/{BASE}"])

    assert not composed.tasks
