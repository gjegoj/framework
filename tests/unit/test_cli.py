"""The entry point: what a command line composes is what the run is built from.

Hydra's own composition is exercised for real — the groups, the example, the overrides — while what it
composes into is stubbed, because everything below has its own tests and none of them needs a fit here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from omegaconf import MissingMandatoryValue

from src import cli
from src.config import ExperimentConfig


def test_the_configs_directory_is_found_by_absolute_path() -> None:
    """Both entry points share it, so a run started from anywhere composes the same groups."""
    assert (Path(cli.CONFIG_DIRECTORY) / "config.yaml").exists()


def test_a_command_line_composes_a_declaration_and_hands_it_to_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started: dict[str, Any] = {}
    monkeypatch.setattr(cli, "build", lambda config: started.setdefault("built", config))
    monkeypatch.setattr(cli, "run", lambda experiment: started.setdefault("ran", experiment))
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "experiment=examples/classification", "lr=0.05", f"hydra.run.dir={tmp_path}"],
    )

    cli.main()

    composed = started["built"]
    assert isinstance(composed, ExperimentConfig)
    assert composed.tasks["label"].target == "species", "the example was composed"
    assert composed.lr == 0.05, "and the override applied"
    assert started["ran"] is composed, "the run is started over the very thing that was built"


def test_a_run_that_never_said_where_its_data_is_is_told_that(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`data.source: ???` is a promise the root config makes; resolving it away would blame the file format."""
    monkeypatch.setattr(cli, "build", lambda config: None)
    # Hydra turns an escaping error into a printed message and exit(1); this asks for the error itself.
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "+tasks.t.kind=classification", "+tasks.t.target=species", f"hydra.run.dir={tmp_path}"],
    )

    with pytest.raises(MissingMandatoryValue, match=r"data\.source"):
        cli.main()
