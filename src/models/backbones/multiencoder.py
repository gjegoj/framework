"""Several networks side by side: one per input a run pairs, each publishing under a name of its own."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from typing import cast

from torch import Tensor

from src.core import TensorShape, TensorTree, as_children
from src.models.base import Backbone
from src.models.registry import backbone_registry

log = logging.getLogger(__name__)

SEPARATOR = "_"
"""What joins a tower's name to the stream it publishes: ``image`` and ``pooled`` become ``image_pooled``.

An underscore rather than a dot, because a stream name becomes the name of a head's child wherever a
head reads several of them, and torch has no child whose name carries a dot.
"""


@backbone_registry.register("multiencoder")
class MultiEncoderBackbone(Backbone):
    """One network per input, encoding side by side into streams that one head can be built over together.

    What a pairing is: a picture read by one family and its caption by another, compared in a space both
    are projected into. The towers share nothing — not their weights, not their width, not a vocabulary —
    so what they have in common is only that a head is built over both, which is why they arrive as
    ordinary feature streams rather than as anything this class invents.

    Every stream is published as ``<tower>_<stream>``. Two towers publish ``pooled`` alike, so the name
    the declaration gave each is the only thing that tells them apart, and a pair of names that would
    join to one word is refused while the run is assembled rather than left to replace one another. A
    run reads them with ``head: {name: linear, stream: [image_pooled, text_pooled]}``.

    Nothing here routes an input to a tower. Each already names the one it reads — ``input_name`` — and
    each is handed the batch whole, so a tower joining a pairing is a line of declaration and no edit.

    It brings no head of its own: which tower a native head would sit on is not something a pairing
    says anywhere, and a run asking for one is told so by name.

    Parameters:
        encoders: One network per name, each declared by ``_target_`` as any nested position is.
    """

    def __init__(self, encoders: Mapping[str, Backbone]) -> None:
        super().__init__()
        if not encoders:
            raise ValueError(
                "'encoders' names the networks a pairing encodes with, and this one pairs nothing: a "
                "backbone publishing no stream at all is one no head can be built over."
            )
        foreign = sorted(name for name, one in encoders.items() if not isinstance(one, Backbone))
        if foreign:
            raise TypeError(
                f"'encoders' holds {', '.join(foreign)}, which name no feature streams: a pairing publishes "
                "what its towers publish, so each of them has to be a Backbone."
            )
        self.encoders = as_children(encoders, label="tower")
        self._say_where_a_carried_classifier_stays(encoders)
        # Called for the refusal it carries. Whether two names join to one word is a fact of the towers
        # alone, and a run should hear it while it is assembled rather than at its first batch.
        self._under_their_towers(lambda tower: tower.feature_shapes)

    @staticmethod
    def _say_where_a_carried_classifier_stays(encoders: Mapping[str, Backbone]) -> None:
        """A tower may start from a trained file; a pairing has no class space to put its classifier in.

        Said here because what a tower prints — ``n were held back`` — is the line an ordinary run
        prints too, and there those rows go on to start a head. Nothing reads them in a pairing, so
        the two runs would otherwise read exactly alike.
        """
        carrying = sorted(name for name, tower in encoders.items() if tower.carried_head)
        if carrying:
            log.info(
                "The weights %s started from carried a classifier, and it is left where they were read: "
                "a head here is built over several feature spaces at once, and which of them those rows "
                "answer for is written nowhere. Every head of this run starts fresh.",
                ", ".join(carrying),
            )

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """What each tower publishes, unchanged but for the name: a head over one is sized by that one."""
        return self._under_their_towers(lambda tower: tower.feature_shapes)

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        return self._under_their_towers(lambda tower: cast("Mapping[str, Tensor]", tower(inputs)))

    def _under_their_towers[T](self, read: Callable[[Backbone], Mapping[str, T]]) -> dict[str, T]:
        """Every tower's answer under ``<tower>_<stream>``, refusing two of them that join to one name.

        One home for the joining, because the shapes a head is sized from and the features it is handed
        are two readings of one naming, and a run whose head found a stream at build and lost it at the
        first batch would be the one thing this framework's names exist to make impossible.
        """
        named = [
            (f"{name}{SEPARATOR}{stream}", value)
            for name, tower in self.encoders.items()
            for stream, value in read(cast("Backbone", tower)).items()
        ]
        repeated = sorted(name for name, count in Counter(name for name, _ in named).items() if count > 1)
        if repeated:
            raise ValueError(
                f"Towers {', '.join(sorted(self.encoders))} publish {', '.join(repeated)} between them more "
                f"than once: a stream is published as `<tower>{SEPARATOR}<stream>`, and these join to one "
                "word, so one tower's features would stand in for another's. Rename a tower."
            )
        return dict(named)
