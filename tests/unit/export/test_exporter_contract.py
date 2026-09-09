"""Every registered exporter, against the port: export a two-input model, load it back, get the same numbers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.core import Batch, Prediction
from src.export import DeployableModel, exporter_registry
from tests.support.fakes import PredictOnlyModel


class TwoInputModel(PredictOnlyModel):
    """An image and a vector feeding two outputs — the arities a one-in, one-out artifact would get wrong."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.mix = torch.nn.Linear(4 + 2, 3)

    def predict(self, batch: Batch) -> Prediction:
        pooled = self.trunk(batch.inputs["image"]).mean(dim=(2, 3))
        mixed = torch.cat([pooled, batch.inputs["vector"]], dim=1)
        return Prediction(outputs={"label": torch.softmax(self.mix(mixed), dim=1), "score": mixed.sum(dim=1)})


def graph() -> DeployableModel:
    model = DeployableModel(TwoInputModel(), ["image", "vector"], ["label", "score"])
    model.eval()
    return model


def example() -> tuple[torch.Tensor, ...]:
    return (torch.randn(2, 3, 8, 8), torch.randn(2, 2))


@pytest.mark.parametrize("name", sorted(map(str, exporter_registry)))
def test_the_artifact_returns_what_the_model_returns(name: str, tmp_path: Path) -> None:
    if name == "tensorrt":
        pytest.importorskip("tensorrt", reason="the engine is built by the library, absent here")
    exporter = exporter_registry.create(name)
    model = graph()
    inputs = example()

    path = exporter.export(model, inputs, tmp_path / "model")
    written = exporter.load(path)(inputs)

    assert path.exists()
    for actual, reference in zip(written, model(*inputs), strict=True):
        assert torch.allclose(actual, reference, atol=1e-4)
