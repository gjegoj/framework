"""The composite family: one backbone encodes, one head per task reads the stream it declared."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from torch import Tensor, nn

from src.core import ModelOutput, Representation, TensorTree, as_children
from src.models.base import Backbone, HeadConnection, Model, Neck, produced_by
from src.models.registry import model_registry


@model_registry.register("composite")
class CompositeModel(Model):
    """Encode once, serve every task from a named stream.

    Heads register as ``heads.<task>``, the backbone as ``backbone`` and a neck as ``neck``: those
    paths are the contract a freeze callback, a checkpoint and a parameter group address, so they are
    part of the design. A neck sits *beside* the backbone rather than around it for exactly that
    reason — wrapped, it moved every path under ``backbone`` one level down, and the recipe
    ``examples/finetuning.yaml`` ships, ``modules: [backbone]``, came to hold the projection still
    along with the encoder it was written for.
    """

    def __init__(self, backbone: Backbone, heads: Mapping[str, HeadConnection], neck: Neck | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        # Measured: an attribute left `None` reaches neither `state_dict` nor `named_children`, so a
        # run that declares no neck writes the checkpoint it wrote before this position existed.
        self.neck = neck
        self.heads = as_children({name: connection.head for name, connection in heads.items()})
        self._streams = {name: connection.streams for name, connection in heads.items()}

    def produces(self, task: str) -> Representation:
        """Whatever the head serving this task says it answers with; one that says nothing projects.

        Every task a composite serves has a head — the build refuses a set of one that is not the set
        of the other — so there is no branch here for a task without one.
        """
        return produced_by(self.heads[task])

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        """A composite gives each task its head and shares everything beneath them; the split is that.

        Everything beneath them, rather than the backbone by name: a neck a run declared is shared by
        every task the same way, and a group named after one of the modules it holds would go on
        reading as though it held only that one.
        """
        return self.heads[task].parameters() if task in self.heads else ()

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = self.backbone(inputs)
        if self.neck is not None:
            features = self.neck(features)
        outputs = {
            name: cast(Tensor, self.heads[name](*(features[stream] for stream in streams)))
            for name, streams in self._streams.items()
        }
        return ModelOutput(outputs=outputs, features=features)
