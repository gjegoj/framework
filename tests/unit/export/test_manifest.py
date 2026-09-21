"""What a deployment is handed beside the artifact: how to build an input, and what an output means."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import onnx
import pytest
import torch
from torch import nn

from src.config import ComponentConfig
from src.core import (
    Axis,
    DatasetInfo,
    InputInfo,
    ModelOutput,
    Normalization,
    TargetInfo,
    TensorShape,
    TensorTree,
    require_tensor,
)
from src.export import DeployableModel
from src.export.base import beside
from src.export.build import build_exporters
from src.export.manifest import MANIFEST_SUFFIX, Manifest, ship
from src.models import Model
from src.tasks import MetricLearning
from src.tasks.classification import Classification
from src.tasks.regression import Regression
from tests.support.models import Angles
from tests.unit.export.test_backends import FEATURES, Heads, deployable, wide

NORMALIZATION = Normalization(mean=(0.485, 0.456, 0.374, 0.5), std=(0.229, 0.224, 0.225, 0.5))


def prepared() -> DatasetInfo:
    """What the run's encoders published about the one input this graph takes."""
    shape = TensorShape(axes=(Axis.CHANNELS,), sizes=(FEATURES,))
    return DatasetInfo(inputs={"features": InputInfo(shape=shape, normalization=NORMALIZATION)}, targets={})


class Publishing(Model):
    """A network whose forward publishes a feature nothing downstream reads — which is what a head with
    hidden widths makes of ``ModelOutput.features``, and what export has to be able to look past."""

    def __init__(self) -> None:
        super().__init__()
        self.answer = nn.Linear(FEATURES, 2)
        self.hidden = nn.Linear(FEATURES, 3)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = require_tensor(inputs["features"], name="features")
        return ModelOutput(
            outputs={"species": self.answer(features)}, features={"species_hidden_0": self.hidden(features)}
        )


def shipped(tmp_path: Path, *declared: dict[str, object], graph: DeployableModel | None = None) -> Manifest:
    formats = build_exporters([ComponentConfig.model_validate(one) for one in declared])
    return ship(graph if graph is not None else deployable(), prepared(), formats, tmp_path / "model")


def test_a_manifest_says_how_to_build_every_input_the_artifact_takes(tmp_path: Path) -> None:
    """Shape, dtype and the scaling the model was trained under — without these a deployment guesses."""
    (built,) = shipped(tmp_path, {"name": "onnx"}).inputs

    assert built.name == "features"
    assert built.shape == (FEATURES,)
    assert built.dtype == "float32"
    assert built.normalization == NORMALIZATION


def test_a_manifest_says_what_every_output_means_in_the_order_the_artifact_answers(tmp_path: Path) -> None:
    """Position is the contract every format keeps; a name is what the numbers in that position mean."""
    outputs = shipped(tmp_path, {"name": "onnx"}).outputs

    assert [one.name for one in outputs] == ["species", "weight"]
    assert outputs[0].semantics == "multiclass"
    assert outputs[0].classes == ("cat", "dog")
    assert outputs[1].semantics is None
    assert outputs[1].classes is None


def test_a_manifest_says_what_an_outputs_numbers_are_and_how_many_of_them_a_row_holds(tmp_path: Path) -> None:
    """A deployment allocating a buffer and reading a position needs both, and a tensor carries neither.

    The shape is one row's, as an input record's is, and it is read off what the graph actually
    answered rather than off what the task declared: reading a projection as what it means changes the
    rank — a score per sample comes back without the axis it was projected along — so `weight` is a
    bare number here while the head serving it produces `[1]`.
    """
    species, weight = shipped(tmp_path, {"name": "onnx"}).outputs

    assert (species.representation, species.shape) == ("probabilities", (2,))
    assert (weight.representation, weight.shape) == ("value", ())


def test_a_record_of_numbers_the_network_already_read_calls_them_by_the_name_they_have(tmp_path: Path) -> None:
    """A bounded reading published under the word `probabilities` is a record of a model nobody built.

    Measured at 37 classes: a converged sample softmaxes to 0.0720 and nothing can exceed 0.1703, so a
    deployment thresholding this artifact at anything at all would reject every picture it is shown —
    while the ranking, which is what the numbers are for, was right the whole time.
    """
    task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))
    network = Angles(task.name, reads="features", in_features=FEATURES, out_features=task.out_features())
    graph = DeployableModel(network, [task], input_names=("features",)).eval()

    (built,) = shipped(tmp_path, {"name": "onnx"}, graph=graph).outputs

    assert built.representation == "cosines"
    assert built.classes == ("cat", "dog") and built.semantics == "multiclass"


def test_a_record_of_a_model_answering_with_a_direction_claims_no_vocabulary(tmp_path: Path) -> None:
    """The identities it was separated by are a training device, and the artifact carries none of them.

    Written as classes, a deployment would read position 0 of a 128-wide unit vector as the first breed
    and index a gallery by it — the record describing a model that was never built.
    """
    identity = MetricLearning("identity", TargetInfo(classes={0: "cat", 1: "dog"}), embedding_dim=FEATURES)
    graph = DeployableModel(
        Heads("features", {identity.name: identity.out_features()}), [identity], input_names=("features",)
    ).eval()

    (built,) = shipped(tmp_path, {"name": "onnx"}, graph=graph).outputs

    assert built.name == "identity"
    assert built.classes is None and built.values is None


def test_a_manifest_says_which_file_is_the_artifact_and_which_only_travel_with_it(tmp_path: Path) -> None:
    """An ONNX model whose weights sit beside it does not travel alone, and copying one file breaks it.

    Named rather than ordered: a deployment has two questions — which file to open, and what has to be
    beside it — and a single list answers neither without knowing that the first entry is the one.
    """
    (built,) = shipped(tmp_path, {"name": "onnx", "external_data": True}, graph=wide()).artifacts

    assert (built.artifact, built.travels_with) == ("model.onnx", ("model.onnx.data",))


def test_an_artifact_that_is_one_file_says_so_by_travelling_with_nothing(tmp_path: Path) -> None:
    """Most formats are one file, and a record promising a second would send a deployment looking for it."""
    (built,) = shipped(tmp_path, {"name": "onnx"}).artifacts

    assert (built.artifact, built.travels_with) == ("model.onnx", ())


def test_a_manifest_records_what_the_file_carries_rather_than_what_was_asked_for(tmp_path: Path) -> None:
    """A declaration may ask for nothing and torch may substitute; the record is read off the artifact."""
    (built,) = shipped(tmp_path, {"name": "onnx"}).artifacts

    assert built.written_by == "OnnxExporter"
    assert isinstance(built.details["opset"], int)


def test_a_manifest_carries_what_each_artifact_was_proven_by(tmp_path: Path) -> None:
    """A record of an artifact nobody compared with the model would be a record of a guess."""
    (built,) = shipped(tmp_path, {"name": "pt2"}).artifacts

    assert built.parity.within_tolerance
    assert built.parity.batches == (2, 1)


def test_the_record_is_written_beside_the_artifacts_and_reads_back_as_a_document(tmp_path: Path) -> None:
    """Beside them because that is where a deployment looks, and as JSON because that is what reads it."""
    manifest = shipped(tmp_path, {"name": "onnx"}, {"name": "pt2"})

    written = json.loads(beside(tmp_path / "model", MANIFEST_SUFFIX).read_text(encoding="utf-8"))

    assert [one["artifact"] for one in written["artifacts"]] == ["model.onnx", "model.pt2"]
    assert written["inputs"][0]["normalization"]["mean"] == list(NORMALIZATION.mean)
    assert written == json.loads(json.dumps(manifest.as_record()))


def test_shipping_leaves_the_model_in_the_state_it_was_handed(tmp_path: Path) -> None:
    """An artifact is written from a model in eval, and the reference implementation left it there —
    encoding the order of a run's own steps as a mutation of somebody else's object."""
    graph = deployable().train()

    ship(graph, prepared(), build_exporters([ComponentConfig.model_validate({"name": "pt2"})]), tmp_path / "model")

    assert graph.training


def test_shipping_leaves_the_model_in_eval_when_that_is_how_it_was_handed_over(tmp_path: Path) -> None:
    """The state restored has to be read off the model, not off the wrapper built around it.

    A wrapper is constructed for the export and ``nn.Module`` starts every instance in training, while
    construction does not propagate anything to the child — so a flag read off the wrapper says
    "training" about a model that was handed over in eval, and hands it back switched on.
    """
    handed = deployable()
    model = handed.model.eval()
    graph = DeployableModel(model, list(handed.tasks), input_names=list(handed.input_names))
    assert graph.training and not model.training, "the wrapper starts in training and the model stayed in eval"

    ship(graph, prepared(), build_exporters([ComponentConfig.model_validate({"name": "pt2"})]), tmp_path / "model")

    assert not model.training


def test_an_input_the_graph_does_not_take_cannot_be_shipped(tmp_path: Path) -> None:
    """The example comes from what the data prepared, so the two declarations have to name one thing."""
    other = DatasetInfo(inputs={"image": InputInfo(shape=TensorShape((Axis.CHANNELS,), (FEATURES,)))}, targets={})

    with pytest.raises(LookupError, match="'features'"):
        ship(deployable(), other, build_exporters([ComponentConfig.model_validate({"name": "pt2"})]), tmp_path / "m")


def test_a_run_that_ships_nothing_writes_no_record(tmp_path: Path) -> None:
    """`export: none` is a declaration, not a gap: there is no artifact, so there is nothing to describe."""
    manifest = ship(deployable(), prepared(), [], tmp_path / "model")

    assert manifest.artifacts == ()
    assert not beside(tmp_path / "model", MANIFEST_SUFFIX).exists()
    assert not list(tmp_path.glob("model.*"))


class Restless(Model):
    """A network answering a little differently every call, so nothing written from it can match it."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(FEATURES, 1)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = require_tensor(inputs["features"], name="features")
        return ModelOutput(outputs={"value": self.head(features) + torch.rand(1)})


def test_a_publication_that_fails_leaves_no_record_standing_over_the_artifact_it_described(
    tmp_path: Path,
) -> None:
    """The record is the claim, and a second publication overwrites the files it was a claim about.

    Artifacts are written straight to their final paths, so re-publishing replaces them before anything
    about them is proven. Measured before this held: the first export succeeded, the second overwrote
    the artifact and failed its parity, and the record of the first stayed beside it — a passing verdict
    describing a file that was no longer there.
    """
    declared: dict[str, object] = {"name": "torchscript"}
    shipped(tmp_path, declared)
    record = beside(tmp_path / "model", MANIFEST_SUFFIX)
    assert record.exists(), "the first publication is what the second one has to not leave standing"

    restless = DeployableModel(Restless(), [Regression("value", TargetInfo())], input_names=("features",))
    with pytest.raises(RuntimeError):
        shipped(tmp_path, declared, graph=restless)

    assert not record.exists()


class Dropping(Model):
    """A network that answers differently in training, which is what makes the state visible at all."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(FEATURES, 1)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = require_tensor(inputs["features"], name="features")
        return ModelOutput(outputs={"value": self.head(self.dropout(features))})


def test_an_artifact_is_written_from_the_model_in_eval_however_it_was_handed_over(tmp_path: Path) -> None:
    """Dropout bakes whichever branch it was in, so a graph written while training is a different model.

    Visible here because the model is also the oracle: written in training, the artifact holds one drawn
    mask and the model draws another, and the parity that proves it fails with numbers nobody can read.
    """
    graph = DeployableModel(Dropping(), [Regression("value", TargetInfo())], input_names=("features",)).train()
    formats = build_exporters([ComponentConfig.model_validate({"name": "pt2"})])

    manifest = ship(graph, prepared(), formats, tmp_path / "model")

    assert manifest.artifacts[0].parity.within_tolerance


def test_a_stream_a_head_publishes_for_a_term_is_not_carried_into_the_artifact(tmp_path: Path) -> None:
    """A hidden width is computed for a distillation term and asked for by nothing a deployment runs, so
    it is dead in the graph — and tracing drops it along with the weights that made it.

    Asserted on the weights the file carries rather than on its outputs: a graph is built from the tasks
    it was handed, so an output nothing serves would be left out whatever the forward did, and a test
    reading only those would pass for a reason other than its name. Measured on torch 2.13: this graph
    holds one `Gemm` and one `Softmax`, and the hidden projection is in neither.
    """
    served = [Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))]
    graph = DeployableModel(Publishing(), served, input_names=("features",)).eval()

    shipped(tmp_path, {"name": "onnx", "opset": 18, "simplify": False}, graph=graph)

    written = onnx.load(str(tmp_path / "model.onnx"))
    assert [one.name for one in written.graph.output] == ["species"]
    assert [one.name for one in written.graph.initializer] == ["model.answer.weight", "model.answer.bias"]
