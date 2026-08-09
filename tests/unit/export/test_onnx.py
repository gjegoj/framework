"""The ONNX backend: a graph torch's modern exporter writes, and onnxruntime runs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.core import Batch, Prediction
from src.export import DeployableModel, OnnxExporter, exporter_registry
from tests.support.fakes import PredictOnlyModel


class TwoTaskModel(PredictOnlyModel):
    """A conv trunk feeding a pooled task and a dense one — two outputs of different rank.

    Wide enough that its weights exceed ONNX's 1024-byte externalisation threshold,
    so the tests exercise the two-file artifact a real model produces rather than a
    self-contained graph that hides the sidecar entirely.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.head = torch.nn.Linear(16, 2)
        self.dense = torch.nn.Conv2d(16, 3, 1)

    def predict(self, batch: Batch) -> Prediction:
        features = self.trunk(batch.inputs["image"])
        return Prediction(
            outputs={
                "label": torch.softmax(self.head(features.mean(dim=(2, 3))), dim=1),
                "mask": torch.sigmoid(self.dense(features)),
            }
        )


def graph() -> DeployableModel:
    model = DeployableModel(TwoTaskModel(), ["image"], ["label", "mask"])
    model.eval()
    return model


def example(batch: int = 2) -> tuple[torch.Tensor, ...]:
    return (torch.randn(batch, 3, 8, 8),)


def test_the_written_artifact_returns_what_the_model_returns(tmp_path: Path) -> None:
    """An artifact that drifts from its source is worse than no artifact: it ships silently."""
    exporter = OnnxExporter()
    model = graph()
    inputs = example()

    path = exporter.export(model, inputs, tmp_path / "model")
    written = exporter.load(path)(inputs)

    expected = model(*inputs)
    assert len(written) == 2
    for actual, reference in zip(written, expected, strict=True):
        assert torch.allclose(actual, reference, atol=1e-5)


def test_the_batch_axis_is_dynamic_so_the_artifact_serves_one_sample(tmp_path: Path) -> None:
    """Measured: without dynamic shapes onnxruntime refuses batch 1 outright — the deployment shape."""
    exporter = OnnxExporter()
    model = graph()

    path = exporter.export(model, example(batch=2), tmp_path / "model")
    single = exporter.load(path)(example(batch=1))

    assert single[0].shape[0] == 1
    assert single[1].shape[0] == 1


def test_the_weights_ride_beside_the_graph_in_a_second_file(tmp_path: Path) -> None:
    """External data is torch's default: a deployment copying only the graph copies no weights."""
    path = OnnxExporter().export(graph(), example(), tmp_path / "model")

    assert path == tmp_path / "model.onnx"
    assert path.with_suffix(".onnx.data").is_file()


def test_an_opset_the_exporter_cannot_deliver_is_refused(tmp_path: Path) -> None:
    """Measured: asking for 17 lands 18 after a failed down-conversion, and torch says nothing loud."""
    with pytest.raises(ValueError, match="opset"):
        OnnxExporter(opset_version=17).export(graph(), example(), tmp_path / "model")


def test_the_deprecated_exporter_path_is_refused_at_construction() -> None:
    """The legacy path cannot read the dynamic shapes this backend builds, and torch is removing it."""
    with pytest.raises(ValueError, match="dynamo"):
        OnnxExporter(dynamo=False)


def sidecar_bytes(path: Path) -> int:
    """The weights riding beside the graph; zero when the graph carries them itself."""
    sidecar = path.parent / f"{path.name}.data"
    return sidecar.stat().st_size if sidecar.exists() else 0


def test_simplifying_rewrites_the_weights_instead_of_appending_to_them(tmp_path: Path) -> None:
    """Measured: ONNX's external-data writer appends, so a rewrite over a live sidecar doubles it.

    The sidecar alone is what discriminates. The file names do not change, and the
    total does not either — the rewritten graph proto is leaner than torch's, which
    on a small model hides a doubled weight file behind a smaller total.
    """
    model = graph()
    inputs = example()

    plain = OnnxExporter().export(model, inputs, tmp_path / "plain" / "model")
    simplified = OnnxExporter(simplify=True).export(model, inputs, tmp_path / "simplified" / "model")

    assert sidecar_bytes(plain) > 0, "the fixture must be wide enough to externalise its weights"
    assert sidecar_bytes(simplified) <= sidecar_bytes(plain)
    written = OnnxExporter().load(simplified)(inputs)
    assert torch.allclose(written[0], model(*inputs)[0], atol=1e-5)


def test_the_format_is_reachable_by_name_with_its_own_knobs() -> None:
    """Config declares a format by name and tunes it by constructor argument; both must work."""
    default = exporter_registry.create("onnx")
    tuned = exporter_registry.create("onnx", opset_version=20, atol=1e-2)

    assert isinstance(default, OnnxExporter)
    assert isinstance(tuned, OnnxExporter)
    assert default.opset_version == 18
    assert tuned.opset_version == 20
    assert tuned.atol == pytest.approx(1e-2)


def test_by_default_the_file_speaks_the_run_s_own_vocabulary(tmp_path: Path) -> None:
    """A deployment config reading 'label' beats one reading 'output_0'; that is why it is the default."""
    exporter = OnnxExporter()

    path = exporter.export(graph(), example(), tmp_path / "model")

    session = exporter.load(path).session
    assert [declared.name for declared in session.get_inputs()] == ["image"]
    assert [written.name for written in session.get_outputs()] == ["label", "mask"]


def test_uniform_names_give_every_model_one_interface(tmp_path: Path) -> None:
    """A serving wrapper that must accept any model needs names that do not vary with the task."""
    exporter = OnnxExporter(tensor_names="uniform")

    path = exporter.export(graph(), example(), tmp_path / "model")

    session = exporter.load(path).session
    assert [declared.name for declared in session.get_inputs()] == ["input"]
    assert [written.name for written in session.get_outputs()] == ["output_0", "output_1"]


def test_a_lone_tensor_carries_no_index(tmp_path: Path) -> None:
    """'input_0' with no 'input_1' beside it is noise; a single tensor is just 'input'."""

    class OneTaskModel(PredictOnlyModel):
        def predict(self, batch: Batch) -> Prediction:
            return Prediction(outputs={"label": batch.inputs["image"] * 2})

    model = DeployableModel(OneTaskModel(), ["image"], ["label"])
    model.eval()
    exporter = OnnxExporter(tensor_names="uniform")

    session = exporter.load(exporter.export(model, (torch.randn(2, 4),), tmp_path / "model")).session

    assert [declared.name for declared in session.get_inputs()] == ["input"]
    assert [written.name for written in session.get_outputs()] == ["output"]


def test_a_scheme_that_is_not_offered_is_refused_with_the_valid_ones() -> None:
    """A plausible name must not quietly fall back to the default and ship the wrong interface.

    ``generic`` is what the reference called this feature, so it is what a user
    arriving from it would write. A wrong *word* rather than a misspelling on
    purpose: this repo's spell-checking hook rewrites a near-miss in place, and it
    once turned this very test green by making its value valid.
    """
    with pytest.raises(ValueError, match="TensorNames must be one of declared, uniform"):
        OnnxExporter(tensor_names="generic")  # type: ignore[arg-type]


def test_renaming_leaves_the_parity_report_speaking_of_tasks(tmp_path: Path) -> None:
    """The report says which task drifted; what the file calls it is a separate, deployment concern."""
    model = graph()
    exporter = OnnxExporter(tensor_names="uniform")

    path = exporter.export(model, example(), tmp_path / "model")
    written = exporter.load(path)(example())

    assert model.output_names == ("label", "mask")
    assert len(written) == 2


def test_an_upstream_knob_this_class_never_declared_still_reaches_torch(tmp_path: Path) -> None:
    """The wrapper doctrine: twenty-five upstream parameters stay reachable without editing it."""
    path = OnnxExporter(external_data=False).export(graph(), example(), tmp_path / "model")

    assert path.is_file()
    assert not path.with_suffix(".onnx.data").exists()
