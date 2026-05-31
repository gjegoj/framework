"""Core enumerations shared across layers."""

from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a training/evaluation run.

    A ``StrEnum`` so members compare equal to their plain-string values
    (``Stage.TRAIN == "train"``) and serialize cleanly into configs/logs.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
