"""``DeployableModel``: the run's model as a graph an exporter can trace."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.core import Batch, Prediction
from src.export import DeployableModel, as_outputs
from tests.support.fakes import PredictOnlyModel


class TwoInputModel(PredictOnlyModel):
    """Doubles input 'a' into task 'left' and triples input 'b' into task 'right'.

    Distinct factors per input and per output, so a test can tell which value
    travelled where instead of only that something came back.
    """

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"left": batch.inputs["a"] * 2, "right": batch.inputs["b"] * 3})


def test_outputs_follow_the_declared_task_order() -> None:
    """The deployed graph's output order is a contract a serving stack indexes by position."""
    graph = DeployableModel(TwoInputModel(), ["a", "b"], ["right", "left"])

    first, second = graph(torch.ones(1), torch.ones(1))

    assert first.item() == pytest.approx(3.0)
    assert second.item() == pytest.approx(2.0)


def test_inputs_reach_the_model_under_their_declared_names() -> None:
    """Positional arguments are named on the way in; getting this backwards swaps two streams silently."""
    graph = DeployableModel(TwoInputModel(), ["b", "a"], ["left", "right"])

    left, right = graph(torch.full((1,), 5.0), torch.full((1,), 7.0))

    assert left.item() == pytest.approx(14.0)  # 'a' is the second argument here, 7 * 2
    assert right.item() == pytest.approx(15.0)  # 'b' is the first argument here, 5 * 3


def test_a_wrong_number_of_inputs_fails_loudly() -> None:
    """Silently zipping a short argument list would export a graph missing a stream."""
    graph = DeployableModel(TwoInputModel(), ["a", "b"], ["left"])

    with pytest.raises(ValueError, match="takes 2 input"):
        graph(torch.ones(1))


def test_the_graph_is_what_torch_can_trace() -> None:
    """The whole reason this class exists: Batch and Prediction do not survive tracing, tensors do."""
    graph = DeployableModel(TwoInputModel(), ["a", "b"], ["left", "right"])
    example = (torch.ones(2), torch.ones(2))

    traced = torch.jit.trace(graph, example)
    written: tuple[Tensor, ...] = traced(*example)

    assert torch.equal(written[0], as_outputs(graph(*example))[0])
    assert torch.equal(written[1], as_outputs(graph(*example))[1])


def test_a_single_task_returns_the_tensor_itself() -> None:
    """Every torch model returns a tensor when it has one output; unpacking a fixed 1-tuple is noise."""
    graph = DeployableModel(TwoInputModel(), ["a", "b"], ["left"])

    written = graph(torch.ones(1), torch.ones(1))

    assert isinstance(written, Tensor)


def test_the_artifact_declares_the_arity_rather_than_choosing_it_at_runtime() -> None:
    """The branch has to fold while the graph is built, or the file would carry a Python condition."""
    example = (torch.ones(2), torch.ones(2))
    single = torch.jit.trace(DeployableModel(TwoInputModel(), ["a", "b"], ["left"]), example)
    both = torch.jit.trace(DeployableModel(TwoInputModel(), ["a", "b"], ["left", "right"]), example)

    assert str(single.forward.schema).endswith("-> Tensor")
    assert str(both.forward.schema).endswith("-> ((Tensor, Tensor))")


def test_outputs_read_by_name_are_a_tuple_whatever_the_arity() -> None:
    """Nothing inside the framework should special-case a single-task run."""
    one = DeployableModel(TwoInputModel(), ["a", "b"], ["left"])
    two = DeployableModel(TwoInputModel(), ["a", "b"], ["left", "right"])

    assert len(as_outputs(one(torch.ones(1), torch.ones(1)))) == 1
    assert len(as_outputs(two(torch.ones(1), torch.ones(1)))) == 2
