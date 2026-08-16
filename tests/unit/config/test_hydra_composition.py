"""The shipped Hydra surface: one declared identity feeds the tracker and the run tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.assembly.callbacks import build_callbacks
from src.callbacks.samples import SampleGrid
from src.config import ExperimentConfig
from src.core.entities import Task
from src.core.taxonomy import Objective, OutputTopology

CONFIGS = Path(__file__).parents[3] / "configs"


def test_the_run_directory_follows_the_runs_identity() -> None:
    """runs/<project>/<name>: the tree is derived from the naming, never named twice."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", overrides=["run.name=probe-run"], return_hydra_config=True)

    assert composed.hydra.run.dir == "runs/my-project/probe-run"


def test_the_clearml_group_reads_the_identity_by_interpolation() -> None:
    """`logger=clearml` alone suffices; project and task arrive from run.*."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", overrides=["logger=clearml", "run.name=probe-run"])

    assert composed.logger.project_name == "my-project"
    assert composed.logger.task_name == "probe-run"


def test_a_declared_tracker_name_wins_over_the_identity() -> None:
    """Overriding is the ordinary OmegaConf merge — no arbitration anywhere in code."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", overrides=["logger=clearml", "logger.project_name=other"])

    assert composed.logger.project_name == "other"


def test_the_samples_group_builds_a_grid_that_undoes_the_runs_own_normalisation() -> None:
    """The one documented way to switch the grid on, exercised end to end.

    ``mean``/``std`` have no default and reach the callback only by interpolation,
    so a group that shipped without resolving them would be a page that mis-colours
    every picture — and the whole path lived in prose until this composed it.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        # `run.directory` reads the live Hydra run dir, which exists only in an app.
        composed = compose(
            config_name="config",
            overrides=[
                "callbacks=samples",
                "run.directory=.",  # reads the live Hydra run dir, which exists only in an app
                "+tasks.label.preset=classification",
                "+tasks.label.target=species",
                "+data.source=table.csv",
            ],
        )
    raw = cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True))
    task = Task(name="label", output_topology=OutputTopology.GLOBAL, objective=Objective.MULTICLASS, metrics={})

    built = build_callbacks(ExperimentConfig(**raw), tasks=[task])

    grid = next(callback for callback in built if isinstance(callback, SampleGrid))
    assert grid._mean.flatten().tolist() == pytest.approx(raw["mean"])
    assert "label" in grid._annotators


def test_both_logging_configs_speak_rich() -> None:
    """Console via RichHandler, the job log also into the run's own directory —
    validated live once (probe app); this pins the composed structure."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        composed = compose(config_name="config", return_hydra_config=True)

    job, own = composed.hydra.job_logging, composed.hydra.hydra_logging
    assert job.handlers.rich["class"] == "rich.logging.RichHandler"
    assert own.handlers.rich["class"] == "rich.logging.RichHandler"
    # Raw, unresolved: hydra.runtime.output_dir exists only once the app runs.
    raw_handlers = cast("dict[str, dict[str, str]]", OmegaConf.to_container(job.handlers, resolve=False))
    assert "hydra.runtime.output_dir" in raw_handlers["file"]["filename"]
    assert job.loggers["clearml.resource_monitor"].handlers  # pre-configured: no double printing
