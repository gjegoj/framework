"""Execution modes; dataset split names remain open strings."""

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
