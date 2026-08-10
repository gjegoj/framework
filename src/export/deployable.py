"""The run's model, reshaped into the one thing a tracer can follow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.core.entities import Batch, require_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.ports import Model


class DeployableModel(nn.Module):
    """The run's model as a deployment graph: tensors in, tensors out.

    Neither ``Batch`` nor ``Prediction`` survives tracing — measured, tracing a
    backbone directly raises "Tracer cannot infer type of Features(...)". This is
    the one place that translates, and it translates through ``Model.predict``,
    so every model family exports the same way and the artifact returns exactly
    what validation compared against.

    Names travel with the graph rather than beside it: an exporter that has to
    name tensors in the file reads them off the module they describe, so the two
    cannot drift.

    Parameters:
        model (Model): The trained model, whichever family built it.
        input_names (Sequence[str]): Batch input names, in the order the exported
            graph takes them positionally.
        output_names (Sequence[str]): Task names, in the order the exported graph
            returns them.
    """

    def __init__(self, model: Model, input_names: Sequence[str], output_names: Sequence[str]) -> None:
        super().__init__()
        if not input_names:
            raise ValueError("A deployable graph needs at least one input name.")
        if not output_names:
            raise ValueError("A deployable graph needs at least one output name.")
        self.model = model
        self.input_names = tuple(input_names)
        self.output_names = tuple(output_names)

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """One tensor per task, and a bare tensor when the run has a single task.

        A one-element tuple would make every consumer of a single-head model
        unpack something that never varies, against the convention every torch
        model already follows. The branch costs nothing in the artifact: it is
        decided while the graph is built, so the written signature is ``Tensor``
        or ``(Tensor, Tensor)`` outright, never a runtime choice.
        """
        if len(inputs) != len(self.input_names):
            declared = ", ".join(self.input_names)
            raise ValueError(f"This graph takes {len(self.input_names)} input(s) ({declared}), got {len(inputs)}.")
        batch = Batch(inputs=dict(zip(self.input_names, inputs, strict=True)), targets={})
        prediction = self.model.predict(batch)
        outputs = tuple(
            require_tensor(prediction.outputs[name], task=name, wanted_by="an exported graph")
            for name in self.output_names
        )
        return outputs[0] if len(outputs) == 1 else outputs


def as_outputs(written: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Whatever a deployable graph returned, as the tuple its output names index.

    The union exists only where deployment ergonomics need it — at the graph's
    own edge. Everything that reads outputs by name goes through here, so a
    single-task run is not a special case anywhere inside.
    """
    return written if isinstance(written, tuple) else (written,)
