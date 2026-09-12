"""A real run to watch: what a callback changes is only visible against one that ran."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightning as L
import pytest
import torch
from torch import Tensor, nn

from src.build import build
from src.config import load_config
from src.experiment import Experiment
from tests.support.declarations import smallest_run
from tests.support.table import write_table


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("rows"))


@pytest.fixture
def declaration(table: Path, tmp_path: Path) -> Mapping[str, Any]:
    """One classification task over eight pictures, and whatever a test declares beside it."""
    return smallest_run(table, tmp_path / "run")


def prepared(declaration: Mapping[str, Any], **overrides: Any) -> Experiment:
    """The run a declaration describes, built but not yet started."""
    return build(load_config({**declaration, **overrides}))


def weights_of(module: nn.Module) -> Tensor:
    """Everything learnable under a module, flattened, so 'did this move' is one comparison."""
    return torch.cat([one.detach().flatten().clone() for one in module.parameters()])


def statistics_of(module: nn.Module) -> Tensor:
    """What a module learned about the data without being optimized — running means, above all."""
    return torch.cat([one.detach().flatten().clone().float() for one in module.buffers()])


def callback_of[T](trainer: L.Trainer, kind: type[T]) -> T:
    """The one callback of a kind a run was given.

    Cast, because Lightning assigns ``callbacks`` in its constructor rather than declaring it on the
    class, so the list a trainer holds carries no type of its own.
    """
    given = cast("list[Any]", trainer.callbacks)  # type: ignore[attr-defined]
    return next(one for one in given if isinstance(one, kind))
