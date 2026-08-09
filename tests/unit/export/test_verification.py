"""Verification: an artifact is not an export until it has been read back and compared."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.core import Batch, Prediction
from src.export import DeployableModel, ExportedArtifact, Exporter, Parity, Runnable, render_report, verify
from tests.support.fakes import PredictOnlyModel


class DoublingModel(PredictOnlyModel):
    """Doubles the 'image' input into task 'label'."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"] * 2})


class FakeExporter(Exporter):
    """Writes nothing and loads back whatever the test says the artifact does.

    Verification's job is to compare a runnable against a model; a fake runnable
    is how a test states the drift it wants compared.
    """

    def __init__(self, runnable: Runnable) -> None:
        super().__init__()
        self._runnable = runnable

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        return destination

    def load(self, path: Path) -> Runnable:
        return self._runnable


def graph() -> DeployableModel:
    return DeployableModel(DoublingModel(), ["image"], ["label"])


def faithful(inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return (inputs[0] * 2,)


def drifted(inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return (inputs[0] * 2 + 0.5,)


def batch_of_two(inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """A graph that baked the batch it was traced at: always two rows, whatever it is fed."""
    return (torch.ones(2, 3) * 2,)


def test_a_faithful_artifact_is_within_tolerance() -> None:
    """The common case must pass, or every export becomes a false alarm."""
    parity = verify(FakeExporter(faithful), Path("unused"), graph(), [(torch.ones(2, 3),)], atol=1e-4, rtol=1e-3)

    assert parity.within_tolerance
    assert parity.max_abs == pytest.approx(0.0)


def test_a_drifted_artifact_is_outside_tolerance() -> None:
    """Silent numerical drift is the failure mode this whole step exists to catch."""
    parity = verify(FakeExporter(drifted), Path("unused"), graph(), [(torch.ones(2, 3),)], atol=1e-4, rtol=1e-3)

    assert not parity.within_tolerance
    assert parity.max_abs == pytest.approx(0.5)
    assert parity.per_output["label"][0] == pytest.approx(0.5)


def test_a_graph_whose_batch_is_baked_is_caught_by_the_second_example() -> None:
    """This is what replaces a topology allowlist: the property is measured, not listed.

    Drop the batch-1 example and this artifact passes — which is the point.
    """
    exporter = FakeExporter(batch_of_two)
    traced_shape = (torch.ones(2, 3),)
    deployment_shape = (torch.ones(1, 3),)

    at_traced_shape = verify(exporter, Path("unused"), graph(), [traced_shape], atol=1e-4, rtol=1e-3)
    at_both_shapes = verify(exporter, Path("unused"), graph(), [traced_shape, deployment_shape], atol=1e-4, rtol=1e-3)

    assert at_traced_shape.within_tolerance
    assert not at_both_shapes.within_tolerance


def test_the_worst_error_of_every_output_is_what_is_reported() -> None:
    """A report that averaged the drift away would hide the one output that broke."""
    parity = Parity(per_output={"a": (0.1, 0.2), "b": (0.3, 0.05)}, within_tolerance=False)

    assert parity.max_abs == pytest.approx(0.3)
    assert parity.max_rel == pytest.approx(0.2)


def test_the_report_names_every_artifact_and_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Drawing is separate from computing, so it can be asserted without capturing a verdict."""
    render_report([ExportedArtifact(path=Path("runs/export/model.pt"), parity=Parity({"label": (0.0, 0.0)}, True))])

    printed = capsys.readouterr().out
    assert "model.pt" in printed
    assert "label" in printed
