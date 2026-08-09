"""The TorchScript backend: what it writes is what the model computes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.core import Batch, Prediction
from src.export import DeployableModel, TorchScriptExporter, as_outputs, exporter_registry
from src.export.backends.torchscript import accelerators
from tests.support.fakes import PredictOnlyModel


class ScaleModel(PredictOnlyModel):
    """Scales the 'image' input into task 'label' — enough graph to trace, small enough to read."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": torch.sigmoid(batch.inputs["image"] * 1.5)})


def graph() -> DeployableModel:
    return DeployableModel(ScaleModel(), ["image"], ["label"])


def test_the_written_artifact_returns_what_the_model_returns(tmp_path: Path) -> None:
    """An artifact that drifts from its source is worse than no artifact: it ships silently."""
    exporter = TorchScriptExporter()
    model = graph()
    example = (torch.randn(2, 3),)

    path = exporter.export(model, example, tmp_path / "model")
    written = exporter.load(path)(example)

    assert torch.allclose(written[0], as_outputs(model(*example))[0], atol=0.0)


def test_a_single_task_artifact_hands_back_a_bare_tensor(tmp_path: Path) -> None:
    """What a serving stack loads must match the convention it already knows: one head, one tensor."""
    path = TorchScriptExporter().export(graph(), (torch.randn(2, 3),), tmp_path / "model")

    loaded = torch.jit.load(str(path))

    assert str(loaded.forward.schema).endswith("-> Tensor")


@pytest.mark.skipif(not accelerators(), reason="device portability can only be measured against a second device")
def test_an_artifact_that_cannot_leave_the_trace_device_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A graph that bakes a constant passes every CPU check and then fails on the deployment device.

    ``torch.jit.trace`` freezes a tensor built inside forward at the trace device;
    ``float64`` is enough to make one unmovable, which is exactly how a timm ViT's
    rotary grid behaves.
    """

    class BakesAConstant(PredictOnlyModel):
        def predict(self, batch: Batch) -> Prediction:
            pinned = torch.ones(3, dtype=torch.float64, device=batch.inputs["image"].device)
            return Prediction(outputs={"label": (batch.inputs["image"].double() * pinned).float()})

    model = DeployableModel(BakesAConstant(), ["image"], ["label"])
    model.eval()

    with caplog.at_level("WARNING"):
        TorchScriptExporter().export(model, (torch.randn(2, 3),), tmp_path / "model")

    assert "refused it" in caplog.text
    assert "dynamic_img_size" in caplog.text


def test_the_returned_path_is_the_file_that_was_written(tmp_path: Path) -> None:
    """The caller verifies and reports what was written, so it must be told, not guess a suffix."""
    path = TorchScriptExporter().export(graph(), (torch.randn(2, 3),), tmp_path / "nested" / "model")

    assert path == tmp_path / "nested" / "model.pt"
    assert path.is_file()


def test_the_format_is_reachable_by_name_with_its_own_tolerances() -> None:
    """Config declares a format by name and tunes it by constructor argument; both must work."""
    default = exporter_registry.create("torchscript")
    tuned = exporter_registry.create("torchscript", atol=1e-2)

    assert isinstance(default, TorchScriptExporter)
    assert default.atol == pytest.approx(1e-4)
    assert default.rtol == pytest.approx(1e-3)
    assert tuned.atol == pytest.approx(1e-2)
