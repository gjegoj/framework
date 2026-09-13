"""The graph a run ships, and the example it is given: tensors in, what each task means out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import Tensor

from src.core import (
    FEATURE_AXIS,
    Axis,
    DatasetInfo,
    InputInfo,
    ModelOutput,
    ShapeTree,
    TargetInfo,
    TensorShape,
    TensorTree,
    require_tensor,
)
from src.export import DeployableModel, example_inputs
from src.models import Model
from src.tasks import Task
from src.tasks.classification import Classification
from src.tasks.regression import Regression

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
IMAGE = TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(3, 8, 8))


class Passes(Model):
    """Answers each task with the input it was told to read, so a test sees which tensor arrived where."""

    def __init__(self, serves: Mapping[str, str]) -> None:
        super().__init__()
        self._serves = dict(serves)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(
            outputs={task: require_tensor(inputs[name], name=name) for task, name in self._serves.items()}
        )


def graph_of(serves: Mapping[str, str], tasks: Sequence[Task]) -> DeployableModel:
    """A graph over a network that hands each task the input it names, in the order the mapping declares."""
    return DeployableModel(Passes(serves), tasks, input_names=tuple(serves.values())).eval()


def two_tasks() -> DeployableModel:
    """Two plain regressions, each reading an input of its own: postprocessing barely changes the value."""
    tasks = [Regression("left", TargetInfo()), Regression("right", TargetInfo())]
    return graph_of({"left": "first", "right": "second"}, tasks)


def test_each_tensor_reaches_the_model_under_the_name_the_graph_takes_it_by() -> None:
    """Positional at the artifact's edge, named inside: the first argument is the first declared input."""
    graph = two_tasks()
    first, second = torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])

    left, right = graph(first, second)

    assert torch.equal(left, first.squeeze(FEATURE_AXIS))
    assert torch.equal(right, second.squeeze(FEATURE_AXIS))


def test_a_call_with_another_number_of_tensors_is_refused_naming_both_counts() -> None:
    """A graph fed the wrong arity would otherwise pair inputs by position and answer about nothing."""
    graph = two_tasks()

    with pytest.raises(ValueError, match="2 expected, 1 given"):
        graph(torch.tensor([[1.0]]))


def test_an_output_means_what_its_task_says_rather_than_what_the_head_produced() -> None:
    """The artifact ships probabilities because the task already states that is what its output means."""
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
    graph = graph_of({"species": "image"}, [Classification("species", TargetInfo(classes=CLASSES))])

    shipped = graph(logits)

    assert torch.allclose(shipped.sum(dim=FEATURE_AXIS), torch.ones(2))
    assert not torch.allclose(shipped, logits)


def test_a_graph_serving_one_task_answers_with_a_tensor_rather_than_a_tuple() -> None:
    """Every torch model with one output does, and a one-element tuple makes deployment unpack nothing."""
    graph = graph_of({"value": "image"}, [Regression("value", TargetInfo())])

    assert isinstance(graph(torch.tensor([[1.0]])), Tensor)


def test_the_outputs_are_named_after_the_tasks_that_produced_them() -> None:
    """A format that writes output names reads them here; a run never writes a second list of them."""
    assert two_tasks().output_names == ("left", "right")


@pytest.mark.parametrize(
    ("serves", "tasks", "expected"),
    [
        ({}, [Regression("value", TargetInfo())], "at least one input"),
        ({"value": "image"}, [], "at least one task"),
    ],
    ids=["no inputs", "no tasks"],
)
def test_a_graph_that_would_carry_nothing_is_refused_at_construction(
    serves: Mapping[str, str], tasks: Sequence[Task], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        DeployableModel(Passes(serves), tasks, input_names=tuple(serves.values()))


def data_declaring(**inputs: InputInfo) -> DatasetInfo:
    return DatasetInfo(inputs=inputs, targets={})


def test_the_example_has_the_shape_the_data_declares_under_the_batch_it_is_asked_for() -> None:
    """Derived from what the encoders published, so export needs no dataset and no second declaration."""
    (example,) = example_inputs(data_declaring(image=InputInfo(shape=IMAGE)), ("image",), batch_size=2)

    assert example.shape == (2, 3, 8, 8)
    assert example.dtype.is_floating_point


def test_the_example_follows_the_order_the_graph_takes_its_inputs() -> None:
    """One home for that order — the graph declares it, the data only sizes it; two would swap silently."""
    info = data_declaring(first=InputInfo(shape=IMAGE), second=InputInfo(shape=TensorShape((Axis.CHANNELS,), (5,))))

    example = example_inputs(info, ("second", "first"), batch_size=1)

    assert [tuple(tensor.shape) for tensor in example] == [(1, 5), (1, 3, 8, 8)]


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, "declares no shape"),
        (TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(3, None, None)), "height, width"),
        ({"left": IMAGE}, "tree of tensors"),
    ],
    ids=["no shape", "an open size", "a tree of tensors"],
)
def test_an_input_whose_shape_is_not_one_known_tensor_is_refused_by_name(declared: ShapeTree, expected: str) -> None:
    """The only real limit on what can be exported, refused where the shape is read rather than deep in torch."""
    info = data_declaring(image=InputInfo(shape=declared))

    with pytest.raises(ValueError, match=f"'image'.*{expected}"):
        example_inputs(info, ("image",), batch_size=1)


def test_an_input_the_data_never_prepared_is_refused_by_name() -> None:
    """A graph over an input the run does not carry has nothing to be given, and says so by name."""
    info = data_declaring(image=InputInfo(shape=IMAGE))

    # Matched on the sentence rather than the name: a bare `KeyError` is a `LookupError` too, and it
    # carries exactly `'photo'` as its message — so the looser assertion would hold with no refusal at all.
    with pytest.raises(LookupError, match="graph takes an input named 'photo'"):
        example_inputs(info, ("photo",), batch_size=1)
