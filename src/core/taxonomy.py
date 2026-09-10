"""Execution modes; dataset split names remain open strings."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


class Direction(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Axis(StrEnum):
    CLASSES = "classes"
    CHANNELS = "channels"
    HEIGHT = "height"
    WIDTH = "width"
    TOKENS = "tokens"
    VIEWS = "views"
    FRAMES = "frames"


class Stream(StrEnum):
    """Conventional feature names; custom names and layer aliases remain valid strings."""

    POOLED = "pooled"
    ENCODER = "encoder"
    DECODER = "decoder"
    EMBEDDINGS = "embeddings"


class Modality(StrEnum):
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    EMBEDDING = "embedding"


class Geometry(StrEnum):
    """How raw values participate in spatial transforms before target encoding."""

    NONE = "none"
    IMAGE = "image"
    MASK = "mask"
    BOXES = "boxes"
