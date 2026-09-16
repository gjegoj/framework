"""A run that adapts: the delta is what learns, and what the run ends holding is named like an ordinary one.

Assembled through the composition root, because the whole point of the phase is an order — the model is
adapted between being built and being trained, and folded back after the epoch the run kept is restored
and before anything is evaluated or shipped. Nothing short of a real run has that order in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.experiment import run
from src.training.checkpoints import MODEL_PREFIX
from tests.support.declarations import smallest_run
from tests.support.table import write_table

ADAPTED = "conv2"
"""Every convolution of a residual block's second position, which is what a low-rank delta is added to here.

The second rather than the first because peft matches a name against the end of a module's path, and
this network's stem is a module named ``conv1`` holding convolutions of its own — a name matching both
a stem and a block is a name peft refuses, and rightly."""

LIVE_BRANCHES = {"zero_init_last": False}
"""Every residual branch starts live rather than at zero, so a delta inside one has a gradient to learn from.

timm zero-initialises the last normalisation of each block, which makes the block the identity until
that one tensor moves. Measured 2026-09-16 over the two optimizer steps this run has: with branches
left at zero, not one block's delta moved at all — under `resnet18` the assertion below was satisfied
by the stem, a convolution outside every branch, and the run proved nothing about the blocks it named.
"""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def declared(table: Path, tmp_path: Path) -> dict[str, Any]:
    """The smallest run there is, with the branches a delta is added to actually carrying signal."""
    declaration = dict(smallest_run(table, tmp_path))
    backbone = dict(declaration["model"]["backbone"])
    declaration["model"] = {**declaration["model"], "backbone": {**backbone, **LIVE_BRANCHES}}
    return declaration


@pytest.fixture
def adapting(declared: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """The same run, adapted, and keeping the epoch it scored best on — the file is written while adapted."""
    return {
        **declared,
        "adapter": {"name": "lora", "module": "backbone", "target_modules": [ADAPTED], "r": 2},
        "callbacks": [{"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(tmp_path / "kept")}],
    }


def state_of(experiment: Any) -> Mapping[str, torch.Tensor]:
    """Everything the network would be written to a file as, which is what a name has to match."""
    return dict(experiment.module.learner.model.state_dict())


def parameters_of(experiment: Any) -> Mapping[str, torch.Tensor]:
    """What a run learns, which is less than what it holds: normalisation keeps statistics, not weights."""
    return dict(experiment.module.learner.model.named_parameters())


def test_what_a_run_that_adapts_ends_holding_is_named_like_a_run_that_never_did(
    declared: dict[str, Any], adapting: dict[str, Any]
) -> None:
    """A checkpoint and an artifact of an adapted run are read by a run that declares no adapter."""
    ordinary = build(load_config(declared))
    adapted = build(load_config(adapting))
    assert list(state_of(adapted)) != list(state_of(ordinary))

    run(adapted)

    assert list(state_of(adapted)) == list(state_of(ordinary))


def test_the_delta_and_the_heads_are_the_only_things_that_moved(
    declared: dict[str, Any], adapting: dict[str, Any]
) -> None:
    """What adapting buys: the weights it was added to are held still, and the fold is what changes them.

    Weights, and only weights. Normalisation goes on collecting this data's statistics while the rest is
    held, exactly as it does under a freeze that declares ``train_bn``; those are buffers rather than
    parameters, and a run that reports otherwise would be reporting on something it did not learn.
    """
    adapted = build(load_config(adapting))
    run(adapted)
    untrained = parameters_of(build(load_config(declared)))

    moved = sorted(
        name
        for name, value in parameters_of(adapted).items()
        if name in untrained and value.dtype.is_floating_point and not torch.equal(value, untrained[name])
    )

    assert [name for name in moved if not name.startswith("heads.") and ADAPTED not in name] == []
    assert [name for name in moved if ADAPTED in name] != []
    assert [name for name in moved if name.startswith("heads.")] != []


def test_the_epoch_a_run_kept_is_restored_before_the_delta_is_folded_in(
    adapting: dict[str, Any], tmp_path: Path
) -> None:
    """The file is written while the network is adapted, so folding first would leave nothing able to read it."""
    adapted = build(load_config(adapting))

    run(adapted)

    kept = sorted((tmp_path / "kept").glob("*.ckpt"))
    assert len(kept) == 1
    written = torch.load(kept[0], map_location="cpu", weights_only=True)["state_dict"]
    under_model = {name.removeprefix(MODEL_PREFIX) for name in written if name.startswith(MODEL_PREFIX)}
    assert under_model != set(state_of(adapted))
