"""Two networks in one run: one teaches from a file, the other learns — and only one of them survives it.

Assembled through the composition root, because what the phase is about is an arrangement rather than a
formula: which network the optimizer sees, which one a checkpoint holds, and which one is shipped.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.experiment import run
from src.training import DistillationLearner
from src.training.checkpoints import model_weights
from tests.support.declarations import smallest_run
from tests.support.table import write_table

SOFT = "train/species/distillation"


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def declared(table: Path, tmp_path: Path) -> dict[str, Any]:
    """A run that keeps the epoch it scored best on and writes its numbers down, so both can be read back."""
    return {
        **smallest_run(table, tmp_path),
        "tracker": {"name": "csv", "save_dir": str(tmp_path / "recorded"), "version": ""},
        "callbacks": [{"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(tmp_path / "kept")}],
    }


def teacher_of(declared: Mapping[str, Any], tmp_path: Path) -> str:
    """An ordinary run, whose kept weights are what the run below learns from."""
    ordinary = {**declared, "callbacks": [{"name": "checkpoint", "dirpath": str(tmp_path / "taught")}]}
    run(build(load_config(ordinary)))
    return str(next(iter(sorted((tmp_path / "taught").glob("*.ckpt")))))


def distilling(declared: Mapping[str, Any], teacher: str, loss: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A run learning from a second network as well as from its targets; `loss` is how far it is from it."""
    return {
        **declared,
        "learner": {
            "name": "distillation",
            "weight": 0.5,
            "loss": loss if loss is not None else {"name": "kullback_leibler", "temperature": 4.0},
        },
        "teacher": {**declared["model"], "checkpoint_path": teacher},
    }


def answering_in_angles(declared: Mapping[str, Any]) -> dict[str, Any]:
    """The same run with a `cosine` head under the angular objective that needs one."""
    task = {**declared["tasks"]["species"], "head": {"name": "cosine", "embedding_dim": 8}}
    return {**declared, "tasks": {"species": {**task, "loss": {"name": "arcface", "margin": 0.5, "scale": 16.0}}}}


def test_the_run_keeps_and_ships_the_student_alone(declared: dict[str, Any], tmp_path: Path) -> None:
    """The second network is a way of learning, not a thing the run produces: nothing it leaves holds it."""
    taught = build(load_config(distilling(declared, teacher_of(declared, tmp_path))))
    ordinary = build(load_config(declared))

    run(taught)

    kept = torch.load(next(iter(sorted((tmp_path / "kept").glob("*.ckpt")))), map_location="cpu", weights_only=True)
    # Against what an ordinary run would have written rather than against a name: a second network
    # registered under any word at all is a key here that an ordinary run does not have.
    assert {name for name in kept["state_dict"] if name.startswith("learner.")} == {
        f"learner.{name}" for name in ordinary.module.learner.state_dict()
    }
    assert list(taught.module.learner.model.state_dict()) == list(ordinary.module.learner.model.state_dict())


def test_the_teacher_answers_with_the_weights_it_was_pointed_at(declared: dict[str, Any], tmp_path: Path) -> None:
    """A network whose head was only just initialised answers with noise, and a run would descend towards it."""
    taught = teacher_of(declared, tmp_path)
    distilled = build(load_config(distilling(declared, taught)))
    learner = distilled.module.learner
    assert isinstance(learner, DistillationLearner)

    held = model_weights(taught)
    answering = learner.teacher.state_dict()

    assert set(answering) == set(held)
    assert all(torch.equal(value, held[name]) for name, value in answering.items())
    untrained = build(load_config(declared)).module.learner.model.state_dict()
    assert any(not torch.equal(value, untrained[name]) for name, value in answering.items())


def test_what_the_student_learned_from_the_teacher_is_a_row_of_its_own(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """A term the total is made of and the report does not show is a number nobody can account for."""
    taught = build(load_config(distilling(declared, teacher_of(declared, tmp_path))))

    run(taught)

    written = sorted((tmp_path / "recorded").rglob("metrics.csv"))
    assert len(written) == 1
    with written[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert SOFT in columns


def test_a_head_that_answers_in_angles_is_distilled_only_where_the_term_is_told_so(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The arrangement this position exists for, and the one it has to refuse — assembled, not simulated.

    A `cosine` head answers in ±1, and a divergence over those numbers is nearly flat whatever the two
    networks think: measured on eight real classes, 256 times smaller than the same comparison made once
    the angles are turned into logits, and every number a run reports about it looks ordinary. Nothing in
    the tensor says which space it holds, so the objective says — and a run that does not say is stopped
    while it is assembled rather than trained on the flat number.
    """
    angular = answering_in_angles(declared)
    teacher = teacher_of(angular, tmp_path)

    with pytest.raises(ValueError, match=r"learner\.loss"):
        build(load_config(distilling(angular, teacher)))

    told = {"name": "kullback_leibler", "temperature": 4.0, "scale": 16.0}
    run(build(load_config(distilling(angular, teacher, told))))

    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        assert SOFT in next(iter(csv.reader(recorded)))
