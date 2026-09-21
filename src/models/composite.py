"""The composite family: one backbone encodes, one head per task reads the stream it declared."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from torch import Tensor, nn

from src.core import ModelOutput, Representation, TensorTree, as_children
from src.models.base import Backbone, HeadConnection, Model, Neck, PublishesStreams, produced_by
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
        """Encode once, answer every task, and publish what each head computed on the way to its answer.

        Two mappings for two things: ``features`` is what the heads read — the encoding half's streams,
        and it does not grow — while ``published`` is what a term of ``learner.loss`` compares, which is
        those and whatever a head added. A head therefore never reads another head's stream by the shape
        of this loop rather than by a refusal written somewhere else.

        A head's stream is filed as ``<task>_<stream>``, the way a tower's is ``image_pooled``: two tasks
        may declare the same head, and a head does not know which of them it was registered under. This
        is the one place that rule is written.

        With ``_`` and not the ``/`` this framework reports under, because a term names the stream and
        reports as ``<stage>/<stream>/representation``, which is read back as ``stage/task/name``:
        measured, ``species_hidden_0`` lands in the task slot exactly as ``pooled`` does — one graph,
        a line per stage, a row in the summary — while ``species/hidden_0`` is read as the task
        ``species`` with a family beneath it, drawn a stage per graph and dropped from the summary.
        """
        features = self.backbone(inputs)
        if self.neck is not None:
            features = self.neck(features)
        outputs: dict[str, Tensor] = {}
        published: dict[str, Tensor] = dict(features)
        for name, streams in self._streams.items():
            read = tuple(features[stream] for stream in streams)
            head = self.heads[name]
            # Asked here rather than once at construction, though the answer cannot change between two
            # batches: hoisting it leaves this line a `cast`, and a cast is the type checker told to stop
            # looking — measured, that is exactly what let the protocol and its one implementer disagree
            # about their signatures unnoticed. The check costs 0.18 us where it holds and 1.45 us where
            # it does not, against the 62 us this head's own forward takes.
            if not isinstance(head, PublishesStreams):
                outputs[name] = cast(Tensor, head(*read))
                continue
            outputs[name], added = head.forward_intermediates(*read)
            published.update({f"{name}_{stream}": value for stream, value in added.items()})
        return ModelOutput(outputs=outputs, features=published)
