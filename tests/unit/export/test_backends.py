"""Every format writes a file that is the graph, reads it back, and says what it actually wrote."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import ModelOutput, TargetInfo, TensorTree, as_children, require_tensor
from src.export import DeployableModel, Exporter, as_outputs
from src.export.backends import ncnn
from src.export.backends.onnx import CONVERTERS
from src.export.registry import exporter_registry
from src.export.verification import verify
from src.models import Model
from src.tasks.classification import Classification
from src.tasks.regression import Regression

FEATURES = 4
WRITTEN_AT = 2
RUNNABLE = ["pt2", "onnx", "torchscript"]
"""The formats this machine can prove.

TensorRT needs a library that is not a dependency, and ncnn a converter binary that will not start here
(it is built for a newer macOS than this one). Their declarations are held to everything that can be
checked without either — including the refusal every format shares, below — and their parity is measured
by nobody, which the milestone report says aloud."""

EVERY = sorted(exporter_registry)
"""Every format a run may declare, including the two nothing here can run."""


class Heads(Model):
    """One linear head per task over one named input: the smallest network an artifact can be made of."""

    def __init__(self, reads: str, widths: Mapping[str, int]) -> None:
        super().__init__()
        self._reads = reads
        self.heads = as_children({name: nn.Linear(FEATURES, width) for name, width in widths.items()})

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = require_tensor(inputs[self._reads], name=self._reads)
        return ModelOutput(outputs={name: head(features) for name, head in self.heads.items()})


def deployable(tasks: int = 2) -> DeployableModel:
    """A graph over two tasks whose outputs differ in shape, or over the first of them alone."""
    served = [
        Classification("species", TargetInfo(classes={0: "cat", 1: "dog"})),
        Regression("weight", TargetInfo()),
    ][:tasks]
    widths = {task.name: task.out_features(task.info) for task in served}
    return DeployableModel(Heads("features", widths), served, input_names=("features",)).eval()


def wide() -> DeployableModel:
    """A graph with weights worth writing beside it: under a kilobyte the library keeps them inside."""
    task = Classification("species", TargetInfo(classes={index: f"class{index}" for index in range(300)}))
    return DeployableModel(
        Heads("features", {task.name: task.out_features(task.info)}), [task], input_names=("features",)
    ).eval()


def example(batch: int = WRITTEN_AT) -> tuple[Tensor, ...]:
    return (torch.randn(batch, FEATURES),)


def exporter(format_name: str, **declared: Any) -> Exporter:
    return exporter_registry.get(format_name)(**declared)


@pytest.mark.parametrize("format_name", RUNNABLE)
def test_the_file_a_format_answers_with_is_the_file_it_wrote(format_name: str, tmp_path: Path) -> None:
    """A format writing more than one file has to say which of them is the artifact."""
    written = exporter(format_name).export(deployable(), example(), tmp_path / "model")

    assert written.exists()
    assert written == exporter(format_name).artifact_path(tmp_path / "model")


@pytest.mark.parametrize("format_name", RUNNABLE)
def test_a_format_makes_room_for_the_file_it_is_about_to_write(format_name: str, tmp_path: Path) -> None:
    """A run declares where its artifacts go, not a directory that already exists for them to go into."""
    written = exporter(format_name).export(deployable(), example(), tmp_path / "artifacts" / "model")

    assert written.exists()


@pytest.mark.parametrize("format_name", RUNNABLE)
@pytest.mark.parametrize("tasks", [1, 2], ids=["one task", "two tasks"])
def test_a_written_artifact_answers_what_the_graph_answers(format_name: str, tasks: int, tmp_path: Path) -> None:
    """The whole point of an artifact: read back, it is still the model the run trained."""
    graph, given = deployable(tasks), example()
    backend = exporter(format_name)

    runnable = backend.load(backend.export(graph, given, tmp_path / "model"))

    with torch.no_grad():
        live = as_outputs(graph(*given))
    read = runnable(given)
    assert len(read) == tasks
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(read, live, strict=True))


@pytest.mark.parametrize("format_name", RUNNABLE)
def test_an_artifact_serves_a_batch_it_was_never_written_at(format_name: str, tmp_path: Path) -> None:
    """The batch axis is declared free rather than hoped to be: a deployment sends one row, not two."""
    graph, backend = deployable(), exporter(format_name)

    runnable = backend.load(backend.export(graph, example(), tmp_path / "model"))

    for batch in (1, 5):
        given = example(batch)
        with torch.no_grad():
            live = as_outputs(graph(*given))
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(runnable(given), live, strict=True))


@pytest.mark.parametrize("format_name", EVERY)
def test_an_example_of_one_row_cannot_carry_a_free_batch_and_is_refused_saying_so(
    format_name: str, tmp_path: Path
) -> None:
    """Held for every format, including the two this machine cannot run: the refusal comes before any of
    their libraries is looked for, which is the only way a declaration answers the same on every machine.

    Measured on torch 2.13: a size-1 axis is settled to one, and torch's own words are "Constraints
    violated (batch)", which name neither what went wrong nor what to do about it."""
    with pytest.raises(ValueError, match="one row"):
        exporter(format_name).export(deployable(), example(1), tmp_path / "model")


def test_the_names_the_graph_declares_are_the_names_inside_the_onnx_file(tmp_path: Path) -> None:
    """Named tensors are what a serving runtime feeds by; nothing else in the tree checks they landed."""
    import onnxruntime

    graph = deployable()
    written = exporter("onnx").export(graph, example(), tmp_path / "model")

    session = onnxruntime.InferenceSession(str(written))

    assert tuple(one.name for one in session.get_inputs()) == graph.input_names
    assert tuple(one.name for one in session.get_outputs()) == graph.output_names


def test_writing_a_graph_keeps_the_converters_own_notes_out_of_the_runs_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Measured on torch 2.13 with onnxscript 0.7.1: six records per artifact about nodes folded and
    removed, arriving over the top of what the run produced. `verbose=False` does not touch them."""
    with caplog.at_level(logging.INFO):
        exporter("onnx").export(deployable(), example(), tmp_path / "model")

    assert [one.name for one in caplog.records if one.name.startswith(CONVERTERS)] == []
    assert [logging.getLogger(name).level for name in CONVERTERS] == [logging.NOTSET] * len(CONVERTERS)


def test_an_opset_the_file_cannot_carry_is_refused_naming_what_was_asked_and_what_was_written(
    tmp_path: Path,
) -> None:
    """Measured on torch 2.13: an unreachable opset is silently replaced by 18, and the file says nothing.

    A run declares an opset because some runtime needs that one, so shipping another under its name is
    the failure this refusal exists for.
    """
    with pytest.raises(ValueError, match=r"opset 9.*18"):
        exporter("onnx", opset=9).export(deployable(), example(), tmp_path / "model")


@pytest.mark.parametrize("separate", [False, True], ids=["one file", "weights beside it"])
def test_where_the_weights_of_an_onnx_model_go_is_the_declaration_that_says_so(separate: bool, tmp_path: Path) -> None:
    """A file whose weights live beside it does not travel alone, so which it is cannot be a surprise.

    Over a graph whose weights are worth separating: measured, under a kilobyte they stay in the file
    however the declaration reads, and the record says which happened rather than which was asked for.
    """
    backend = exporter("onnx", external_data=separate)

    written = backend.export(wide(), example(), tmp_path / "model")

    assert Path(f"{written}.data").exists() is separate
    assert backend.travels_with(written) == ((Path(f"{written}.data"),) if separate else ())


def test_what_an_artifact_records_is_read_off_that_file_rather_than_off_a_declaration(tmp_path: Path) -> None:
    """torch substitutes an operator set it cannot convert a model to, so a record built from what was
    asked for would name a version the file does not carry — which is what a runtime would then refuse."""
    written = exporter("onnx", opset=17).export(deployable(), example(), tmp_path / "model")

    assert exporter("onnx", opset=20).describe(written)["opset"] == 17


@pytest.mark.parametrize("folded", [False, True], ids=["as torch wrote it", "folded afterwards"])
def test_an_onnx_record_says_whether_the_graph_was_folded(folded: bool, tmp_path: Path) -> None:
    """A deployment comparing two artifacts of one model needs to know which passes each has been through."""
    backend = exporter("onnx", simplify=folded)

    written = backend.export(deployable(), example(), tmp_path / "model")

    assert backend.describe(written)["simplified"] is folded


def test_an_ncnn_graph_and_its_weights_are_one_artifact_under_one_name(tmp_path: Path) -> None:
    """A `.param` says what the network is and a `.bin` holds it; either one alone deploys nothing."""
    graph = exporter("ncnn")
    written = graph.artifact_path(tmp_path / "model.v2")

    assert (written.name, graph.weights_of(written).name) == ("model.v2.param", "model.v2.bin")  # type: ignore[attr-defined]


def test_the_converter_is_told_where_to_write_in_terms_that_do_not_depend_on_where_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pnnx is run inside a scratch directory so that its own by-products land there and are swept up.

    The destination is the caller's, and the shipped one is relative — `hydra.run.dir` is
    `runs/${run.project}/${run.name}` — so a path handed over as written resolves against the scratch
    directory instead, and the artifact is deleted with it. The converter would have done its job, exited
    zero, and the refusal would name pnnx.
    """
    seen: list[str] = []

    def converter(argv: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        """A stand-in that does what the tool does: writes where it is told, from where it is run."""
        told = dict(one.split("=", 1) for one in argv[1:] if "=" in one)
        for key in ("ncnnparam", "ncnnbin"):
            seen.append(told[key])
            (Path(options["cwd"]) / told[key]).write_bytes(b"")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(ncnn.subprocess, "run", converter)
    monkeypatch.chdir(tmp_path)

    written = exporter("ncnn").export(deployable(), example(), Path("runs/a-run/model"))

    assert all(Path(one).is_absolute() for one in seen), seen
    assert (tmp_path / written).exists()


def test_a_graph_ncnn_will_not_read_is_refused_rather_than_crashing_the_process(tmp_path: Path) -> None:
    """Measured: ncnn answers -1 and carries on, and the next call into the net segfaults the run."""
    junk = tmp_path / "model.param"
    junk.write_text("not a graph at all\n")
    (tmp_path / "model.bin").write_bytes(b"")

    with pytest.raises(ValueError, match=r"model\.param"):
        exporter("ncnn").load(junk)


def test_folding_what_is_constant_makes_a_smaller_file_that_is_still_the_model(tmp_path: Path) -> None:
    """Another library rewrites the file torch wrote, so it is held to the same proof, and to having done
    something at all. The specimen here is untrained, which is where folding has most to take: measured,
    the same resnet18 loses 28.3% of its bytes untrained and 0.3% trained."""
    graph, backend = deployable(), exporter("onnx", simplify=True)
    plain = exporter("onnx", simplify=False).export(graph, example(), tmp_path / "plain")

    written = backend.export(graph, example(), tmp_path / "folded")

    assert written.stat().st_size < plain.stat().st_size
    assert verify(backend, written, graph, example()).within_tolerance


def test_a_rewrite_its_own_library_cannot_confirm_is_refused_rather_than_shipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """onnxsim checks its own output and says whether it agreed; what torch wrote is the model, and an
    artifact nobody stands behind is not an improvement on it."""
    import onnxsim

    monkeypatch.setattr(onnxsim, "simplify", lambda model: (model, False))

    with pytest.raises(RuntimeError, match="simplify: false"):
        exporter("onnx", simplify=True).export(deployable(), example(), tmp_path / "model")


def test_folding_does_not_double_the_weights_written_beside_the_graph(tmp_path: Path) -> None:
    """Measured on the reference implementation: ONNX's external-data writer *appends* to a location
    file that already exists, and writing over a live one grew a 44.70 MB artifact to 76.84 MB, half of
    it unreachable. The sidecar goes before the rewrite, and a declared pair stays a pair."""
    plain = exporter("onnx", external_data=True, simplify=False).export(wide(), example(), tmp_path / "plain")
    folded = exporter("onnx", external_data=True, simplify=True).export(wide(), example(), tmp_path / "folded")

    assert Path(f"{folded}.data").exists()
    assert Path(f"{folded}.data").stat().st_size <= Path(f"{plain}.data").stat().st_size


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ({"precision": "int4"}, "int4"),
        ({"min_batch": 0}, "0 < min"),
        ({"min_batch": 4, "opt_batch": 2}, "min 4"),
        ({"opt_batch": 8, "max_batch": 4}, "opt 8"),
    ],
    ids=["an unknown precision", "a batch of none", "an optimum below the minimum", "an optimum above the maximum"],
)
def test_a_tensorrt_engine_that_could_not_be_built_is_refused_before_the_library_is_looked_for(
    declared: Mapping[str, Any], expected: str
) -> None:
    """The library is on no machine here, and a typo has to answer on every machine all the same."""
    with pytest.raises(ValueError, match=expected):
        exporter("tensorrt", **declared)


@pytest.mark.parametrize("asked", ["export", "load"], ids=["writing", "reading"])
def test_without_the_library_tensorrt_refuses_by_naming_it(asked: str, tmp_path: Path) -> None:
    """Neither half of the contract pretends: an engine cannot be written or read without the runtime."""
    backend = exporter("tensorrt")
    calls = {
        "export": lambda: backend.export(deployable(), example(), tmp_path / "model"),
        "load": lambda: backend.load(tmp_path / "model.engine"),
    }

    with pytest.raises(ImportError, match="tensorrt"):
        calls[asked]()
