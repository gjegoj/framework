"""The export section, turned into exporters and the example they are given."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.assembly.export import build_exporters, example_inputs, export_model
from src.core import Batch, Prediction
from src.export import TorchScriptExporter
from tests.support.configs import DATA, paper_config
from tests.support.fakes import PredictOnlyModel


class PoolingModel(PredictOnlyModel):
    """Averages an image into one number per sample — a model with a shape opinion."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"].mean(dim=(1, 2, 3), keepdim=False)})


class VectorModel(PredictOnlyModel):
    """Reads a flat vector, so an image-shaped example cannot reach it."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"] @ torch.ones(7)})


def test_the_example_shape_follows_image_size_and_the_normalisation_channels() -> None:
    """Config must never restate a fact it already carries; the shape is read, not declared."""
    config = paper_config(image_size=[32, 48], mean=[0.5])

    example = example_inputs(config, 2)

    assert len(example) == 1
    assert tuple(example[0].shape) == (2, 1, 32, 48)


def test_one_example_tensor_is_built_for_every_declared_input() -> None:
    """A dual-encoder graph takes two tensors; building one would trace half a model."""
    config = paper_config(data=DATA | {"inputs": {"image": {"column": "image"}, "sketch": {"column": "sketch"}}})

    assert len(example_inputs(config, 1)) == 2


def test_the_declared_formats_are_built_in_order() -> None:
    """The export section is a list because the order it is written in is the order it runs."""
    exporters = build_exporters(paper_config(export=[{"name": "torchscript", "atol": 0.01}]))

    assert len(exporters) == 1
    assert isinstance(exporters[0], TorchScriptExporter)
    assert exporters[0].atol == pytest.approx(0.01)


def test_no_export_section_builds_nothing() -> None:
    """A run that declares no format must not pay for the phase, nor fail in it."""
    assert build_exporters(paper_config()) == []


def test_two_targets_of_one_format_are_refused_by_name() -> None:
    """Both would write one file; the second would silently overwrite the first."""
    with pytest.raises(ValueError, match="torchscript"):
        build_exporters(paper_config(export=[{"name": "torchscript"}, {"name": "torchscript", "atol": 0.1}]))


def test_an_unknown_format_is_refused_before_training() -> None:
    """Failing at assembly beats failing an hour into a run, and the message lists what exists."""
    with pytest.raises(LookupError, match="torchscript"):
        build_exporters(paper_config(export=[{"name": "torchscrpt"}]))


def test_an_example_the_model_rejects_names_the_shape_and_where_it_came_from(tmp_path: Path) -> None:
    """A raw torch shape error would not tell the user which config fields to change."""
    config = paper_config(run={"directory": str(tmp_path)})

    with pytest.raises(ValueError, match="image_size"):
        export_model(VectorModel(), config, [TorchScriptExporter()])


def test_a_faithful_model_exports_and_reports_its_artifact(tmp_path: Path) -> None:
    """The whole phase, end to end: written under the run's own directory, and verified."""
    config = paper_config(image_size=[8, 8], run={"directory": str(tmp_path)})

    artifacts = export_model(PoolingModel(), config, [TorchScriptExporter()])

    assert len(artifacts) == 1
    assert artifacts[0].path == tmp_path / "export" / "model.pt"
    assert artifacts[0].parity.within_tolerance
