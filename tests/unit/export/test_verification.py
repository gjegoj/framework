"""An artifact is proven against the model it was written from, on every batch it has to serve."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import Tensor

from src.core import ModelOutput, TargetInfo, TensorTree, require_tensor
from src.export import DeployableModel, Exporter, Runnable, as_outputs
from src.export.backends.tensorrt import TensorRtExporter
from src.export.verification import Parity, verify
from src.models import Model
from src.tasks.regression import Regression
from tests.unit.export.test_backends import RUNNABLE, deployable, example, exporter

WRITTEN_AT = 2


class Fake(Exporter):
    """An artifact that answers however a test needs it to, so verification is what is under test."""

    suffix: ClassVar[str] = "fake"

    def __init__(self, answer: Runnable, *, atol: float = 1e-4, rtol: float = 0.0) -> None:
        super().__init__(atol=atol, rtol=rtol)
        self._answer = answer

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        path.write_bytes(b"")

    def load(self, path: Path) -> Runnable:
        return self._answer


def truthfully(graph: DeployableModel, *, off_by: float = 0.0) -> Runnable:
    """An artifact that is the model, give or take a fixed amount on every value it answers with."""

    def answer(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            return tuple(one + off_by for one in as_outputs(graph(*tensors)))

    return answer


def proven(graph: DeployableModel, answer: Runnable, tmp_path: Path, **tolerances: float) -> Parity:
    backend = Fake(answer, **tolerances)
    return verify(backend, backend.export(graph, example(WRITTEN_AT), tmp_path / "model"), graph, example(WRITTEN_AT))


@pytest.mark.parametrize("format_name", RUNNABLE)
def test_an_artifact_that_is_the_model_is_proven_on_every_batch_it_has_to_serve(
    format_name: str, tmp_path: Path
) -> None:
    """The real thing, end to end: written, read back by its own runtime, and compared with the model."""
    graph, backend = deployable(), exporter(format_name)
    written = backend.export(graph, example(WRITTEN_AT), tmp_path / "model")

    parity = verify(backend, written, graph, example(WRITTEN_AT))

    assert parity.within_tolerance
    assert parity.batches == (WRITTEN_AT, 1)


def test_an_artifact_that_drifted_is_refused_naming_it_and_how_far(tmp_path: Path) -> None:
    """A file that is not the model is worse than no file: it looks shippable and answers differently.

    Both numbers are spelled out, because a looser pattern is met by either of them alone: ``2.5`` is a
    substring of ``2.500e-04``, so a test asking only for that would pass however the share was computed.
    """
    graph = deployable()

    with pytest.raises(RuntimeError, match=r"model\.fake.*disagrees by 2\.500e-04, which is 2\.5 times"):
        proven(graph, truthfully(graph, off_by=2.5e-4), tmp_path)


def test_how_far_an_artifact_stands_from_the_model_is_measured_and_not_merely_judged(tmp_path: Path) -> None:
    """The record carries the drift itself, and it is the only number in it read in the values' own units —
    a verdict alone would let an artifact sit one part in a thousand from the model with nothing to show it."""
    graph = deployable()

    parity = proven(graph, truthfully(graph, off_by=0.6e-4), tmp_path)

    assert parity.difference == pytest.approx(0.6e-4, rel=1e-3)
    assert parity.allowance_used == pytest.approx(0.6, rel=1e-3)


class Scaled(Model):
    """Two constant outputs of very different size, so one element's allowance dwarfs the other's."""

    BIG: ClassVar[float] = 1000.0

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        rows = require_tensor(inputs["features"], name="features").sum(-1, keepdim=True) * 0.0
        return ModelOutput(outputs={"big": rows + self.BIG, "small": rows})


def test_the_difference_reported_is_the_one_at_the_element_that_used_the_allowance(tmp_path: Path) -> None:
    """Two independent maxima let the refusal state an arithmetic impossibility.

    A relative allowance is wide where the value is large: here 0.5 off a thousand is half of what is
    allowed, while 1e-3 off zero is ten times it. Taking the largest gap and the largest share separately
    reports the first number and the second verdict, joined by "which is" — a sentence that is not true of
    any element. The gap carried has to be the one at the element the verdict was made from.
    """
    graph = DeployableModel(
        Scaled(), [Regression("big", TargetInfo()), Regression("small", TargetInfo())], input_names=("features",)
    )

    def answer(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            big, small = as_outputs(graph(*tensors))
        return (big + 0.5, small + 1e-3)

    with pytest.raises(RuntimeError, match=r"disagrees by 1\.000e-03, which is 10\.0 times"):
        proven(graph, answer, tmp_path, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    ("off_by", "holds"), [(0.9e-4, True), (1.1e-4, False)], ids=["inside the allowance", "beyond it"]
)
def test_an_artifact_is_proven_exactly_while_it_stays_inside_the_allowance(
    off_by: float, holds: bool, tmp_path: Path
) -> None:
    """Where the line is, measured through the same formula the reported number comes from."""
    graph = deployable()

    if holds:
        assert proven(graph, truthfully(graph, off_by=off_by), tmp_path).within_tolerance
    else:
        with pytest.raises(RuntimeError):
            proven(graph, truthfully(graph, off_by=off_by), tmp_path)


@pytest.mark.parametrize(("used", "holds"), [(0.0, True), (1.0, True), (1.0001, False)])
def test_the_number_a_parity_reports_is_the_one_its_verdict_is_made_of(used: float, holds: bool) -> None:
    """The reference implementation printed a relative error computed one way and judged by another, so a
    green verdict could stand beside a number that looked catastrophic. One quantity answers both."""
    assert Parity(allowance_used=used, difference=0.0, batches=(2,)).within_tolerance is holds


def test_an_artifact_stuck_at_the_batch_it_was_written_from_is_caught_by_the_other_size(
    tmp_path: Path,
) -> None:
    """Why a second size is asked at all: an artifact that settled its batch answers the written one
    perfectly, and a deployment sending a single row is the first to find out."""
    graph = deployable()
    with torch.no_grad():
        settled = as_outputs(graph(*example(WRITTEN_AT)))

    with pytest.raises(RuntimeError, match="species"):
        proven(graph, lambda tensors: settled, tmp_path)


def only_the_first_output_drifts(graph: DeployableModel) -> Runnable:
    def answer(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            written = as_outputs(graph(*tensors))
        return (written[0] + 2.5e-4, *written[1:])

    return answer


def only_the_written_batch_drifts(graph: DeployableModel) -> Runnable:
    def answer(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            written = as_outputs(graph(*tensors))
        off_by = 2.5e-4 if tensors[0].shape[0] == WRITTEN_AT else 0.0
        return tuple(one + off_by for one in written)

    return answer


@pytest.mark.parametrize(
    "drifting", [only_the_first_output_drifts, only_the_written_batch_drifts], ids=["one output", "one batch"]
)
def test_the_verdict_is_made_of_the_worst_answer_rather_than_the_last_one(
    drifting: Callable[[DeployableModel], Runnable], tmp_path: Path
) -> None:
    """One task out of two, or one batch size out of two, is one artifact that is not the model — and
    each is checked in a loop of its own, so either could keep only what it saw last."""
    graph = deployable()

    with pytest.raises(RuntimeError, match=r"model\.fake"):
        proven(graph, drifting(graph), tmp_path)


def test_an_output_of_another_shape_is_refused_rather_than_quietly_broadcast(tmp_path: Path) -> None:
    """Subtracting `[batch, 1]` from `[batch, 2]` gives a small difference and a green verdict."""
    graph = deployable()

    def narrower(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            written = as_outputs(graph(*tensors))
        return (written[0][:, :1], *written[1:])

    with pytest.raises(RuntimeError, match="species"):
        proven(graph, narrower, tmp_path)


def test_an_artifact_answering_with_another_number_of_outputs_is_refused_naming_both_counts(
    tmp_path: Path,
) -> None:
    graph = deployable()

    def fewer(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            return as_outputs(graph(*tensors))[:1]

    with pytest.raises(RuntimeError, match=r"2 outputs.*with 1"):
        proven(graph, fewer, tmp_path)


def test_an_engine_is_proven_across_the_batch_profile_it_was_built_for() -> None:
    """A format whose artifact serves a declared range says so; the default is what any run deploys at."""
    engine = TensorRtExporter(min_batch=2, opt_batch=4, max_batch=8)

    assert engine.answers_at(written_at=4) == (2, 4, 8)
