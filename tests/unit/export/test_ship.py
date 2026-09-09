"""The export phase: every declared format is written from the weights in memory, and proven."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.export import DeployableModel, TorchScriptExporter
from src.export.ship import ExampleRejected, ship
from tests.support.fakes import DoublingModel, FakeExporter, VectorModel


def image(batch: int) -> tuple[Tensor, ...]:
    return (torch.randn(batch, 3, 8, 8),)


def test_a_faithful_graph_is_written_under_the_destination_and_verified(tmp_path: Path) -> None:
    graph = DeployableModel(DoublingModel(), ["image"], ["label"])

    artifacts = ship(graph, image, [TorchScriptExporter()], tmp_path / "export" / "model")

    assert [artifact.path for artifact in artifacts] == [tmp_path / "export" / "model.pt"]
    assert artifacts[0].parity.within_tolerance


def test_an_example_the_graph_rejects_is_refused_before_any_format_is_written(tmp_path: Path) -> None:
    """One forward, negligible beside tracing the whole graph, turns a raw torch error into a named refusal."""
    graph = DeployableModel(VectorModel(), ["image"], ["label"])

    with pytest.raises(ExampleRejected, match=r"image \(2, 3, 8, 8\)"):
        ship(graph, image, [TorchScriptExporter()], tmp_path / "export" / "model")

    assert not (tmp_path / "export").exists()


def test_an_artifact_that_drifted_is_refused_naming_the_file(tmp_path: Path) -> None:
    """A written file that is not the model must not be reported as an export."""
    graph = DeployableModel(DoublingModel(), ["image"], ["label"])
    drifting = FakeExporter(lambda inputs: (inputs[0] * 2 + 0.5,))

    with pytest.raises(RuntimeError, match="model"):
        ship(graph, image, [drifting], tmp_path / "export" / "model")
