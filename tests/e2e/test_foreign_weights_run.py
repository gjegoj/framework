"""A run starting from weights of its own architecture trained elsewhere, for a class space that has grown.

Assembled through the composition root, because what the phase is about is an order and a hand-off: the
backbone is filled while it is built, holds back the classifier it has no place for, and the head this
run declares is built around those rows — three steps in three modules that only a real run puts in line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import timm
import torch

from src.build import build
from src.config import load_config
from src.core import require_tensor, submodule_at
from src.experiment import run
from src.models import Model
from src.models.heads import ExpandedHead
from tests.support.declarations import BACKBONE, CLASSES, smallest_run
from tests.support.table import write_table

if TYPE_CHECKING:
    from pathlib import Path

GROWN = {**CLASSES, len(CLASSES): "bird"}
"""The vocabulary of a later run: every name the file answered for, at its index, and one added after."""

FIRST_WEIGHT = "conv1.0.weight"
"""The first tensor of the library's own graph, named as the library names it — a stem convolution here.

Read as a witness that a tensor of the file landed in the network rather than beside it, so any real
weight would serve; this one is named because the framework prefixes it and the file does not."""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file naming the architecture, as a hub hosts one and as a pretraining script writes one.

    Weights of the library's own graph, classifier included — the shape that has nothing to do with how
    this framework wraps that graph, which is why the backbone is the one asked where it keeps it.
    """
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / f"{BACKBONE}.pth"
    torch.save(timm.create_model(BACKBONE, pretrained=False, num_classes=len(CLASSES)).state_dict(), path)
    return path


def written_in(path: Path) -> dict[str, torch.Tensor]:
    """What the file holds, read the way anything but this framework would read it."""
    return dict(torch.load(path, map_location="cpu", weights_only=True))


@pytest.fixture
def starting(table: Path, tmp_path: Path, trained: Path) -> dict[str, Any]:
    declared = dict(smallest_run(table, tmp_path))
    declared["model"] = {
        "name": "composite",
        "backbone": {"name": "timm", "model_name": BACKBONE, "pretrained": False, "checkpoint_path": str(trained)},
    }
    declared["tasks"] = {"species": {"kind": "classification", "target_column": "species", "classes": GROWN}}
    return declared


def network_of(experiment: Any) -> Model:
    """The network a run holds, as what it is, so the parts a declaration names can be read back."""
    model = experiment.module.learner.model
    assert isinstance(model, Model)
    return model


def test_the_classes_the_file_answered_for_keep_their_rows_and_the_one_added_since_is_fresh(
    starting: dict[str, Any], trained: Path
) -> None:
    """The declared vocabulary pins a name to a position, so the carried rows mean what they meant."""
    written = written_in(trained)

    held = dict(network_of(build(load_config(starting))).state_dict())

    assert torch.equal(held[f"backbone.model.{FIRST_WEIGHT}"], written[FIRST_WEIGHT])
    assert torch.equal(held["heads.species.base.projection.weight"], written["fc.weight"])
    assert held["heads.species.novel.projection.weight"].shape == (1, written["fc.weight"].shape[1])


def test_what_was_carried_and_what_is_new_are_two_paths_a_declaration_can_hold_apart(
    starting: dict[str, Any],
) -> None:
    """`freeze`, a parameter group and a checkpoint all address parts by path; growing adds two, not a shape."""
    model = network_of(build(load_config(starting)))

    assert isinstance(submodule_at(model, "heads.species", reader="test"), ExpandedHead)
    assert {"heads.species.base.projection.weight", "heads.species.novel.projection.weight"} <= set(model.state_dict())


def test_a_run_that_started_from_a_file_trains_and_answers_for_every_class_it_declared(
    starting: dict[str, Any],
) -> None:
    """The whole of it: the run goes through, and the grown head answers on the class axis for all of them."""
    experiment = build(load_config(starting))

    run(experiment)

    answered = network_of(experiment)({"image": torch.zeros(2, 3, 32, 32)}).outputs["species"]
    assert require_tensor(answered, name="species").shape == (2, len(GROWN))
