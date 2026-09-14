"""What a deployment is handed beside the artifact: how to build an input, and what an output means."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.core import DatasetInfo, Normalization, class_name
from src.export.base import BATCH_AXIS, WRITTEN_FROM, Exporter, beside
from src.export.deployable import DeployableModel, as_outputs, example_inputs
from src.export.verification import Parity, verify

MANIFEST_SUFFIX = "json"
"""What the record is written under, beside the artifacts it describes and named as they are."""


@dataclass(frozen=True, slots=True)
class InputRecord:
    """One tensor a deployment has to build, and everything it takes to build it.

    ``shape`` is one sample's. The batch is the axis in front of it, and how many rows a given artifact
    will take is that artifact's own business, recorded with it below.

    ``normalization`` is what the input's encoder declares, and nothing at build compares it against
    what the stage chain applies. It is true because the two are not written twice: a shipped chain
    interpolates ``${preprocessing.inputs.image.mean}`` rather than repeating the numbers, so there is
    one declaration and nothing to disagree with it. A run spelling its own numbers into its own chain
    is the one case this record can be wrong about, and writing them once is what keeps it right.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    normalization: Normalization | None


@dataclass(frozen=True, slots=True)
class OutputRecord:
    """One tensor a deployment reads back, and what the numbers in it mean.

    ``representation`` is what the numbers themselves are — a share per class, a number in the target's
    own units, a direction, an angle — which no tensor carries and which decides whether a deployment
    may threshold them at all. ``shape`` is one row's, as an input record's is, and is read off what
    the graph actually answered: reading a projection as what it means changes the rank, so a shape
    worked out from what the head produces would be a second statement about the same file.

    ``classes`` is the vocabulary in index order, which is what turns a position in the tensor into a
    word; ``values`` is what each position stands for where a number was learned as a distribution over
    bins. A task that means neither carries no vocabulary and needs none: a plain number, and an
    embedding, whose positions are a direction and stand for nothing one at a time.
    """

    name: str
    representation: str
    shape: tuple[int, ...]
    semantics: str | None
    classes: tuple[str, ...] | None
    values: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One written artifact: the files it is made of, what wrote it, and what it was proven against.

    A record of an artifact nobody compared with the model would be a record of a guess, so the parity
    travels with it — including the batch sizes it was asked at, because that is the claim being made.

    ``artifact`` is the file a deployment opens and ``travels_with`` what has to be beside it. Two names
    rather than one list, because those are two questions and a list answers them only to a reader who
    already knows that the first entry is the one.
    """

    artifact: str
    travels_with: tuple[str, ...]
    written_by: str
    details: Mapping[str, object]
    parity: Parity


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything a deployment needs that it cannot work out from an artifact by itself.

    Outputs are listed in the order every format answers in, and that order is the contract: a format
    that carries names writes these same ones, and one that names its own blobs — ncnn does — is read by
    position. Nothing here is written twice: the inputs come from what the encoders published once the
    run's preprocessing was fitted, and the outputs from the tasks the run declared.
    """

    inputs: tuple[InputRecord, ...]
    outputs: tuple[OutputRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]

    def as_record(self) -> dict[str, Any]:
        """The same manifest as plain data, which is the shape a file and a tracker both take.

        Named for what receives it — ``KeepsRecord.log_record`` — so one object does not pick up a third
        word between the package that builds it and the one that keeps it.
        """
        return asdict(self)


def ship(graph: DeployableModel, info: DatasetInfo, exporters: Sequence[Exporter], destination: Path) -> Manifest:
    """Write every declared format, prove each one against the model, and describe them all.

    The record is the deliverable, not a side effect: without it an artifact carries its weights and
    nothing else — not the scaling its inputs were trained under, not what the numbers coming out of it
    stand for — and every deployment rediscovers those by reading the training code.

    One example serves everything: it is what each format is written from and what each is verified on,
    so no two of them can be describing different tensors.
    """
    if not exporters:
        # Before the example, not after: a run that ships nothing has no reason to be refused by an
        # input whose shape it was never going to write.
        return Manifest(inputs=(), outputs=(), artifacts=())
    example = example_inputs(info, graph.input_names, WRITTEN_FROM)
    with _as_it_is_shipped(graph):
        answered = _answered(graph, example)
        artifacts = tuple(_written(exporter, graph, example, destination) for exporter in exporters)
    manifest = Manifest(inputs=_inputs(info, graph, example), outputs=_outputs(graph, answered), artifacts=artifacts)
    beside(destination, MANIFEST_SUFFIX).write_text(json.dumps(manifest.as_record(), indent=2), encoding="utf-8")
    return manifest


@contextmanager
def _as_it_is_shipped(graph: DeployableModel) -> Iterator[None]:
    """The graph in the state an artifact is written from, and the caller's own state back afterwards.

    Eval, because every format bakes whichever branch dropout and batch norm were in, and the model is
    also the oracle that proves the artifact — so a graph left training disagrees with its own artifact
    and the parity says so in numbers nobody can read. On CPU, because that is the portable place to
    write from: a traced graph pins constants computed inside ``forward`` to the device it saw.

    The reference implementation called ``eval()`` and ``cpu()`` on the caller's model and left it there,
    which encodes the order of a run's own steps as a mutation of somebody else's object.

    The mode is read off the *model*, not off the graph around it: a run builds that wrapper for the
    export, ``nn.Module`` starts every instance in training, and construction propagates nothing to the
    child — so the wrapper's own flag says "training" about a model handed over in eval, and restoring
    from it would switch the caller's model back on. Reading the child is also right for a graph the
    caller built, since ``train`` and ``eval`` propagate and the two agree wherever anyone has set them.
    """
    parameter = next(graph.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    training = graph.model.training
    try:
        graph.eval().to("cpu")
        yield
    finally:
        graph.to(device).train(training)


def _written(
    exporter: Exporter, graph: DeployableModel, example: tuple[Tensor, ...], destination: Path
) -> ArtifactRecord:
    path = exporter.export(graph, example, destination)
    parity = verify(exporter, path, graph, example)
    return ArtifactRecord(
        artifact=path.name,
        travels_with=tuple(one.name for one in exporter.travels_with(path)),
        written_by=type(exporter).__name__,
        details=dict(exporter.describe(path)),
        parity=parity,
    )


def _inputs(info: DatasetInfo, graph: DeployableModel, example: tuple[Tensor, ...]) -> tuple[InputRecord, ...]:
    """Each input as the artifacts actually take it, read off the example every one of them was written from."""
    return tuple(
        InputRecord(
            name=name,
            shape=_one_row(tensor),
            dtype=str(tensor.dtype).removeprefix("torch."),
            normalization=info.inputs[name].normalization,
        )
        for name, tensor in zip(graph.input_names, example, strict=True)
    )


def _one_row(tensor: Tensor) -> tuple[int, ...]:
    """What one row of a tensor measures, which is what both sides of a record are written in.

    The batch is the axis in front of it: how many rows a given artifact takes is that artifact's own
    business, recorded with it rather than with the tensor it reads or answers with.
    """
    return tuple(int(size) for size in tensor.shape[BATCH_AXIS + 1 :])


def _answered(graph: DeployableModel, example: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """What the graph itself answers to the example every artifact was written from.

    A pass of its own rather than the oracle's inside ``verify``: threading one out of the other would
    tie what the record says about an output to how many formats the run happened to declare.
    """
    with torch.no_grad():
        return as_outputs(graph(*example))


def _outputs(graph: DeployableModel, answered: tuple[Tensor, ...]) -> tuple[OutputRecord, ...]:
    """What each position of the answer means — and, for a direction, that it means no position at all.

    Read off what the task publishes rather than off what its target held: a task answering with an
    embedding may still have been *trained* against a vocabulary, and writing that here would tell a
    deployment to read the first number of a unit vector as the first identity.
    """
    return tuple(
        OutputRecord(
            name=task.name,
            representation=task.answers_with(produced),
            shape=_one_row(tensor),
            semantics=task.semantics,
            classes=None if task.embeds else _vocabulary(task.info.classes),
            values=None if task.embeds else task.info.values,
        )
        for task, produced, tensor in zip(graph.tasks, graph.produced, answered, strict=True)
    )


def _vocabulary(classes: Mapping[int, str] | None) -> tuple[str, ...] | None:
    """A vocabulary as a deployment reads it: the words in index order, so a position is a name.

    A mapping would survive to the file with its keys turned into strings, and a reader would have to
    put them back in order to use them — while the order is already guaranteed, since class indices are
    contiguous from zero wherever a vocabulary is declared.
    """
    return None if classes is None else tuple(class_name(classes, index) for index in range(len(classes)))
