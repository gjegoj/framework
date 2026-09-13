"""The closed vocabularies every package shares, and nothing that is open to a declaration.

What a run is doing, what an axis is, what a stream carries, what a label means, what an input is made
of, which side of a sample a value rides on, and how it moves through a pixel chain. Dataset split names
are not here: a run names its own splits, and only the three that are also stages are spoken for.
"""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """What a run is doing, and — by convention — the name of the split it reads while doing it.

    The convention is what lets a declaration written per stage (`transforms`) reach the split a data
    module prepared, so the two words stay one word wherever they meet.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Axis(StrEnum):
    CLASSES = "classes"
    CHANNELS = "channels"
    HEIGHT = "height"
    WIDTH = "width"
    EMBEDDING = "embedding"


SPATIAL = frozenset({Axis.HEIGHT, Axis.WIDTH})
"""The axes that give a value an extent of its own: a task with one decides at every pixel.

Read from both sides of one rule — a task asks whether it is dense, a head is sized by the one axis of
an output that is *not* one of these.
"""


class Stream(StrEnum):
    """Conventional feature names; custom names and layer aliases remain valid strings."""

    POOLED = "pooled"
    ENCODER = "encoder"
    DECODER = "decoder"


class Semantics(StrEnum):
    """What a label means, independent of where it sits: one class, one score, or one score per label.

    Every library spells this distinction its own way — torchmetrics calls it ``task``, smp calls it
    ``mode`` — so the framework keeps a word of its own and each adapter translates it where that
    library is imported. A task states it once; its loss and its metrics are sized from it.
    """

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Representation(StrEnum):
    """What a head's numbers are, which the shape they arrive in does not say.

    Read where two separately built parts have to agree about one tensor. An angular margin is added to
    an angle and has nothing to add to a projection; an ordinary cross-entropy needs the opposite, and
    over values bounded by one it can never become confident. Neither mistake shows in a shape, and
    neither shows reliably in the values either — measured, a projection whose weights start small
    reaches 0.34, well inside the range a cosine occupies.

    ``PROJECTED`` is the feature mapped and nothing more: logits where the output axis is classes, the
    coordinates of a direction where it is an embedding. ``COSINES`` is bounded to ±1 — the angle
    between the feature and one prototype per class, both read as directions alone.
    """

    PROJECTED = "projected"
    COSINES = "cosines"


class Modality(StrEnum):
    """What an input is; conventional names, as ``Stream``'s are — a custom encoder may say anything."""

    IMAGE = "image"


class Role(StrEnum):
    """The three places a sample carries values the pipeline moves, named once for both sides of it.

    They are ``Sample``'s own field names, which is what lets a preprocessor's ``geometries`` reach
    ``with_geometry`` as keyword arguments without either side spelling them out again.
    """

    INPUTS = "inputs"
    TARGETS = "targets"
    AUXILIARY = "auxiliary_inputs"


class Geometry(StrEnum):
    """How raw values participate in spatial transforms before target encoding."""

    NONE = "none"
    IMAGE = "image"
    MASK = "mask"
