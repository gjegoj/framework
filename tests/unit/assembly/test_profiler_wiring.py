"""The profiler is a component like any other, and its report belongs to the run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from lightning.pytorch.profilers import AdvancedProfiler, PassThroughProfiler, SimpleProfiler

from src.assembly.training import build_trainer
from src.config import ExperimentConfig, TrainerConfig
from tests.support.configs import paper_config

PRESET = Path("configs/trainer/profile.yaml")


def experiment(profiler: Any = None, **run: Any) -> ExperimentConfig:
    """The base experiment with the profiler under test on its trainer, and the run it names."""
    trainer: dict[str, Any] = {"max_epochs": 1}
    if profiler is not None:
        trainer["profiler"] = profiler
    return paper_config(trainer=trainer, run=run)


def profiler_of(declaration: Any = None, **run: Any) -> Any:
    """The profiler a config produced. ``Trainer.profiler`` is assigned at construction
    rather than declared, so it is read through here once instead of ignored at every
    call site.
    """
    return build_trainer(experiment(declaration, **run)).profiler  # type: ignore[attr-defined]


def test_a_named_profiler_reaches_the_trainer_built() -> None:
    """The section forwards its knobs verbatim, and Lightning takes an object here.

    Handed the mapping unbuilt it would reach `Trainer(profiler={...})` and be
    accepted as neither a profiler nor an alias.
    """
    built = profiler_of({"name": "simple"})

    assert isinstance(built, SimpleProfiler)


def test_a_run_that_asks_for_no_profiler_is_not_measured() -> None:
    """Timing every hook is not free, so it happens only where it was asked for."""
    built = profiler_of()

    assert isinstance(built, PassThroughProfiler)


def test_a_profilers_own_arguments_reach_it() -> None:
    """A registry entry that could not be configured would be worth less than an import path."""
    built = profiler_of({"name": "advanced", "line_count_restriction": 5.0})

    assert isinstance(built, AdvancedProfiler)
    assert built.line_count_restriction == 5.0


def test_a_profiler_outside_the_registry_is_reachable_by_import_path() -> None:
    """The registry is a convenience, not a gate — the same sentence as for schedulers."""
    built = profiler_of({"_target_": "lightning.pytorch.profilers.AdvancedProfiler"})

    assert isinstance(built, AdvancedProfiler)


def test_where_the_report_goes_is_whatever_config_said() -> None:
    """Nothing in assembly decides this — the preset writes `${run.directory}`, and
    `test_run_directory.py` is what holds the shipped groups to it.
    """
    built = profiler_of({"name": "simple", "dirpath": "runs/pets/one", "filename": "profile"})

    assert (built.dirpath, built.filename) == ("runs/pets/one", "profile")


def test_a_profiler_lightning_knows_but_we_do_not_is_refused_with_the_ones_we_have() -> None:
    """`xla` is the name to try it with: Lightning's docs offer it, this registry does not.

    Copied from those docs it has to fail by naming what is here, or the reader is
    left with an alias that looks registered and a run that dies at construction on
    a missing `torch_xla` — a message about the wrong problem.
    """
    with pytest.raises(LookupError, match="simple"):
        build_trainer(experiment({"name": "xla"}))


def test_the_shipped_preset_inherits_the_default_trainer_rather_than_restating_it() -> None:
    """A preset that copies its neighbour drifts from it, and the reference's had.

    Its copy pinned `precision: 32-true` while the default trained in `bf16-mixed`,
    so the profiling run measured a program nobody was running. Inherited, there is
    nothing to drift: the preset may say only which profiler.
    """
    content = yaml.safe_load(PRESET.read_text(encoding="utf-8"))

    assert content["defaults"] == ["default"]
    assert set(content) == {"defaults", "profiler"}
    assert set(content["profiler"]) == {"name", "dirpath", "filename"}


def test_the_shipped_preset_names_a_profiler_that_builds() -> None:
    """The file a user swaps in must hold a real declaration, not an example that drifts."""
    content = yaml.safe_load(PRESET.read_text(encoding="utf-8"))

    section = TrainerConfig.model_validate({"profiler": content["profiler"]})

    assert section.profiler is not None
    assert isinstance(profiler_of(content["profiler"]), SimpleProfiler)
