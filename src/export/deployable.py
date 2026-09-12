"""The run's model as the one thing a tracer can follow, and the example it is followed on."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import Tensor, nn

from src.core import DatasetInfo, TensorShape, TensorTree, require_tensor
from src.models import Model
from src.tasks import Task


class DeployableModel(nn.Module):
    """The trained model with this framework's vocabulary taken off both of its ends: tensors in, tensors out.

    An artifact carries no ``Mapping``, no ``ModelOutput`` and no ``Task``, so the translation happens
    here — once, for every model family and every format — and the names travel with the graph, where a
    backend that writes them reads them off the module itself.

    What comes out is what each task *means* — probabilities, a value, a mask — because ``postprocess``
    already states that, and a deployment left to apply a softmax of its own would be a second home for
    a decision the run has made. The outputs come in the order the tasks were declared and are named
    after them, so neither order nor names are written down a second time.

    Measured on torch 2.13: ``torch.jit.script`` cannot compile this, and for reasons that are v2's own
    contracts rather than anything this module could avoid. ``Model.forward`` is annotated
    ``Mapping[str, TensorTree]``, which the compiler rejects outright ("Unknown type constructor
    Mapping", and the recursive tree alias is beyond it as well), and a graph whose arity follows the
    run's declared inputs needs ``*tensors``, which it refuses too. Tracing is therefore the only route
    every backend has, which is what makes a second batch size worth verifying: a traced graph can bake
    one in.
    """

    def __init__(self, model: Model, tasks: Sequence[Task], input_names: Sequence[str]) -> None:
        super().__init__()
        if not input_names:
            raise ValueError("A deployable graph is called with one tensor per input, so it needs at least one input.")
        if not tasks:
            raise ValueError("A deployable graph answers with one tensor per task, so it needs at least one task.")
        self.model = model
        self.tasks = tuple(tasks)
        self.input_names = tuple(input_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """What this graph's outputs are called: the run's task names, in the order it serves them."""
        return tuple(task.name for task in self.tasks)

    def forward(self, *tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
        """One tensor per task, and a bare tensor where the run serves a single one.

        A one-element tuple would make every consumer of a single-task artifact unpack something that
        never varies, against the convention every torch model already follows. The branch costs the
        artifact nothing: it is taken while the graph is traced, so what gets written is one signature
        or the other, never a choice made at inference.
        """
        if len(tensors) != len(self.input_names):
            taken = ", ".join(self.input_names)
            raise ValueError(
                f"This graph is called with one tensor per input, in the order {taken}: "
                f"{len(self.input_names)} expected, {len(tensors)} given."
            )
        inputs = cast(Mapping[str, TensorTree], dict(zip(self.input_names, tensors, strict=True)))
        output = self.model(inputs)
        shipped = tuple(require_tensor(task.postprocess(output), name=task.name) for task in self.tasks)
        return shipped[0] if len(shipped) == 1 else shipped


def as_outputs(written: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Whatever a deployable graph answered, as the tuple its output names index.

    The union exists only at the graph's own edge, where a single-task model answers the way every torch
    model with one output does. Here rather than beside the formats that call it, because it is the other
    half of the branch above: one module decides that a single task answers bare and undoes it for
    everyone reading outputs back, so one task is a special case in one place rather than in each format.
    """
    return written if isinstance(written, tuple) else (written,)


def example_inputs(info: DatasetInfo, names: Sequence[str], batch_size: int) -> tuple[Tensor, ...]:
    """One example tensor per input the graph takes, shaped as the prepared data declares that input.

    Derived rather than declared again: the sizes are the ones the encoders publish once preprocessing
    is fitted, so an example cannot disagree with what the model was trained on, and a run that changed
    its image size needs no second edit anywhere.

    The two halves come from the two places that own them — the graph says which inputs it takes and in
    what order, ``info`` says how large each of them is — so neither has an order to keep in step with
    the other.

    Values are drawn from a normal distribution, which is what a normalized input looks like. A shape is
    all the data facts carry, so an input that is not floating point is refused by the model when it is
    given this example rather than by anything here.
    """
    return tuple(torch.randn(batch_size, *_sizes_of(info, name)) for name in names)


def _sizes_of(info: DatasetInfo, name: str) -> tuple[int, ...]:
    """What one input measures per sample, or a refusal naming the input and what it actually declared."""
    if name not in info.inputs:
        prepared = ", ".join(sorted(info.inputs)) or "nothing"
        raise LookupError(
            f"The graph takes an input named {name!r}, and this run prepared {prepared}. Both names come "
            "from `data.inputs`, so a graph over another input has nothing to be given."
        )
    declared = info.inputs[name].shape
    if declared is None:
        reason = "declares no shape"
    elif not isinstance(declared, TensorShape):
        reason = "is a tree of tensors rather than one"
    elif open_axes := [axis for axis, size in zip(declared.axes, declared.sizes, strict=True) if size is None]:
        reason = f"leaves {', '.join(open_axes)} open"
    else:
        return cast(tuple[int, ...], declared.sizes)
    raise ValueError(
        f"An export gives the graph one example tensor per input, shaped as the data declares it, and "
        f"{name!r} {reason}. An encoder publishes this in the `info` it declares for its input."
    )
