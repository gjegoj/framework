"""The tools beside the framework, run the way a reader runs them: as modules, from the repository root.

As subprocesses on purpose, and not only because these are scripts: a test importing the module would
never see the failure that running one by path gives, which is the failure ``scripts/cut_head_tail.py``
explains. What is checked here is the command a reader types.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.models.weights import weights_in
from tests.support.scripts import cut_head_tail

TASK = "species"
PREFIX = f"learner.model.heads.{TASK}."


def run_checkpoint(at: Path, head: dict[str, Tensor]) -> Path:
    """A run's own file as this framework writes one: the whole training module, one head inside it."""
    written = at / "run.ckpt"
    torch.save({"state_dict": {f"{PREFIX}{name}": value for name, value in head.items()}}, written)
    return written


def test_a_head_of_one_projection_is_cut_whole(tmp_path: Path) -> None:
    """Which is the whole of what a student carrying its teacher's head needs, and what `linear` writes.

    Measured before this held: the tool read every layer path as a numbered one and died on
    `int('projection')` — a stack trace in place of a file, for the most ordinary head there is.
    """
    checkpoint = run_checkpoint(tmp_path, {"projection.weight": torch.ones(8, 128), "projection.bias": torch.zeros(8)})
    output = tmp_path / "head.ckpt"

    finished = cut_head_tail(str(checkpoint), str(output), "--task", TASK)

    assert finished.returncode == 0, finished.stderr
    held = weights_in(str(output))
    assert {name: list(value.shape) for name, value in held.items()} == {
        "projection.weight": [8, 128],
        "projection.bias": [8],
    }


def test_the_tail_of_a_stack_is_renumbered_as_the_students_own_head_numbers_it(tmp_path: Path) -> None:
    """A teacher's `mlp` of three projections, of which the student carries the last two as its own.

    `Mlp` puts a nonlinearity between projections, so a stack of three is `layers.0, 2, 4` and its tail
    of two is `layers.0, 2`: the names change even though the weights do not, which is the whole reason
    this is a tool rather than a copy.
    """
    teacher = {
        "layers.0.weight": torch.ones(64, 256),
        "layers.0.bias": torch.zeros(64),
        "layers.2.weight": torch.full((32, 64), 2.0),
        "layers.2.bias": torch.zeros(32),
        "layers.4.weight": torch.full((8, 32), 3.0),
        "layers.4.bias": torch.zeros(8),
    }
    checkpoint = run_checkpoint(tmp_path, teacher)
    output = tmp_path / "tail.ckpt"

    finished = cut_head_tail(
        str(checkpoint), str(output), "--task", TASK, "--in-features", "64", "--out-features", "8", "--hidden", "32"
    )

    assert finished.returncode == 0, finished.stderr
    held = weights_in(str(output))
    assert sorted(held) == ["layers.0.bias", "layers.0.weight", "layers.2.bias", "layers.2.weight"]
    assert torch.equal(held["layers.0.weight"], teacher["layers.2.weight"])
    assert torch.equal(held["layers.2.weight"], teacher["layers.4.weight"])


@pytest.mark.parametrize(
    ("arguments", "refused_with"),
    [
        pytest.param(("--task", "breed"), "holds no head for 'breed'", id="a task the file never held"),
        pytest.param(
            ("--task", TASK, "--hidden", "32"),
            "in-features",
            id="a tail asked for without the widths it is built at",
        ),
    ],
)
def test_a_cut_that_could_not_be_made_is_refused_by_name(
    tmp_path: Path, arguments: tuple[str, ...], refused_with: str
) -> None:
    """A tool writing nothing says why; a stack trace names neither the file nor the way out."""
    checkpoint = run_checkpoint(tmp_path, {"projection.weight": torch.ones(8, 128), "projection.bias": torch.zeros(8)})

    finished = cut_head_tail(str(checkpoint), str(tmp_path / "out.ckpt"), *arguments)

    assert finished.returncode != 0
    assert refused_with in finished.stderr


def test_a_head_whose_layers_are_not_numbered_is_taken_whole(tmp_path: Path) -> None:
    """A `cosine` head holds `prototypes` beside its projection, and neither of the two is a layer of a
    stack: what makes a head cuttable is the numbering, not how many paths it happens to hold.

    Measured before this held: two unnumbered paths were read as a stack, and the tool exited on
    `'projection' is not a numbered layer` — after writing the file, so it left an output behind and
    then called itself a failure.
    """
    angular = {"projection.weight": torch.ones(4, 16), "prototypes": torch.ones(8, 4)}
    checkpoint = run_checkpoint(tmp_path, angular)
    output = tmp_path / "head.ckpt"

    finished = cut_head_tail(str(checkpoint), str(output), "--task", TASK)

    assert finished.returncode == 0, finished.stderr
    assert sorted(weights_in(str(output))) == ["projection.weight", "prototypes"]


def test_a_tail_of_a_head_that_is_no_stack_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    """Only a stack has a tail, and a head that numbers nothing is one whole thing — which this tool
    takes whole, so the refusal names that way out rather than only the layer it could not read.

    Nothing written: a file left behind by a run that exited non-zero is one a later command reads as
    though the cut had been made.
    """
    checkpoint = run_checkpoint(tmp_path, {"projection.weight": torch.ones(8, 4), "prototypes": torch.ones(8, 4)})
    output = tmp_path / "tail.ckpt"

    finished = cut_head_tail(
        str(checkpoint), str(output), "--task", TASK, "--in-features", "4", "--out-features", "8", "--hidden", "2"
    )

    assert finished.returncode != 0
    assert "whole" in finished.stderr
    assert not output.exists()
