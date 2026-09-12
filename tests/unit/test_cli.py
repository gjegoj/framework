"""The entry point: what a command line composes is what the run is built from.

Hydra's own composition is exercised for real — the groups, the example, the overrides — while what it
composes into is stubbed, because everything below has its own tests and none of them needs a fit here.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from omegaconf import MissingMandatoryValue
from rich.console import Console

from src import cli
from src.config import ExperimentConfig
from src.export import Manifest
from src.export.manifest import ArtifactRecord
from src.export.verification import Parity


def _ran(started: dict[str, Any], experiment: Any) -> Manifest:
    """A run answers with what it shipped; this one ships nothing, which is what `export: none` means."""
    started.setdefault("ran", experiment)
    return Manifest(inputs=(), outputs=(), artifacts=())


def drawn(manifest: Manifest) -> str:
    screen = Console(width=200, record=True)
    screen.print(cli.table_for(manifest))
    return screen.export_text()


def test_what_a_run_shipped_is_shown_with_how_far_each_artifact_stood_from_the_model() -> None:
    """The legacy report's job, done in the layer that may draw: the export owns no presentation, and a
    verdict that existed only as printed text could not be asserted or sent anywhere else."""
    parity = Parity(allowance_used=0.25, difference=3.5e-5, batches=(2, 1))
    artifact = ArtifactRecord(
        artifact="model.onnx",
        travels_with=("model.onnx.data",),
        written_by="OnnxExporter",
        details={},
        parity=parity,
    )

    shown = drawn(Manifest(inputs=(), outputs=(), artifacts=(artifact,)))

    assert "model.onnx" in shown
    assert "model.onnx.data" in shown, "what it does not travel without"
    assert "2, 1" in shown, "the batch sizes it was proven at"
    assert "3.50e-05" in shown
    assert "25.0%" in shown, "the share of its allowance, which is what the refusal is made of"


def test_every_number_is_shown_under_the_column_that_names_it() -> None:
    """Asserting that a value appears *somewhere* in the render cannot tell one column from another, so
    the two numbers could be printed under each other's heading and every such assertion would hold."""
    artifact = ArtifactRecord(
        artifact="model.onnx",
        travels_with=("model.onnx.data",),
        written_by="OnnxExporter",
        details={},
        parity=Parity(allowance_used=0.25, difference=3.5e-5, batches=(2, 1)),
    )

    built = cli.table_for(Manifest(inputs=(), outputs=(), artifacts=(artifact,)))

    assert {one.header: list(one.cells) for one in built.columns} == {
        "Artifact": ["model.onnx"],
        "Travels with": ["model.onnx.data"],
        "Proven at": ["2, 1"],
        "Worst difference": ["3.50e-05"],
        "Of its allowance": ["25.0%"],
    }


def test_an_artifact_that_travels_alone_says_so_rather_than_leaving_a_blank() -> None:
    artifact = ArtifactRecord(
        artifact="model.pt2", travels_with=(), written_by="Pt2Exporter", details={}, parity=Parity(0.0, 0.0, (2, 1))
    )

    assert "nothing" in drawn(Manifest(inputs=(), outputs=(), artifacts=(artifact,)))


def test_a_run_that_shipped_something_ends_by_showing_what(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The table is the point of drawing it here: a run that wrote artifacts says so where a person is.

    Through a console of its own rather than the captured stream: rich's global one is shared with the
    libraries a run is built on, and what it is pointed at depends on whatever ran before this.
    """
    screen = Console(width=200, record=True, file=StringIO())
    monkeypatch.setattr(cli, "console", lambda: screen)
    artifact = ArtifactRecord(
        artifact="model.onnx", travels_with=(), written_by="OnnxExporter", details={}, parity=Parity(0.1, 1e-6, (2, 1))
    )
    monkeypatch.setattr(cli, "build", lambda config: None)
    monkeypatch.setattr(cli, "run", lambda experiment: Manifest(inputs=(), outputs=(), artifacts=(artifact,)))
    monkeypatch.setattr(sys, "argv", ["main.py", "experiment=examples/classification", f"hydra.run.dir={tmp_path}"])

    cli.main()

    assert "model.onnx" in screen.export_text()


def test_the_configs_directory_is_found_by_absolute_path() -> None:
    """Both entry points share it, so a run started from anywhere composes the same groups — and a
    relative path satisfies the file check too, as long as the tests happen to run from the repository."""
    assert Path(cli.CONFIG_DIRECTORY).is_absolute()
    assert (Path(cli.CONFIG_DIRECTORY) / "config.yaml").exists()


def test_a_command_line_composes_a_declaration_and_hands_it_to_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started: dict[str, Any] = {}
    monkeypatch.setattr(cli, "build", lambda config: started.setdefault("built", config))
    monkeypatch.setattr(cli, "run", lambda experiment: _ran(started, experiment))
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "experiment=examples/classification", "lr=0.05", f"hydra.run.dir={tmp_path}"],
    )

    cli.main()

    composed = started["built"]
    assert isinstance(composed, ExperimentConfig)
    assert composed.tasks["label"].target_column == "species", "the example was composed"
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
        ["main.py", "+tasks.t.kind=classification", "+tasks.t.target_column=species", f"hydra.run.dir={tmp_path}"],
    )

    with pytest.raises(MissingMandatoryValue, match=r"data\.source"):
        cli.main()
