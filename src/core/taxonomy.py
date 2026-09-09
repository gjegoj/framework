"""Domain vocabulary: stages, the shape of a prediction, feature streams, geometry and the names of inputs."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Phase of a run a component is operating in."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class OutputTopology(StrEnum):
    """Output structure of a task — what one prediction looks like.

    A closed set: readers that branch on the shape of an output — a batch transform, a
    page — ask this, so a new member changes what the framework can express. What a task
    *learns* is its kind's business, not a second axis here.

    Attributes:
        GLOBAL: One prediction vector per sample (classification, regression, embeddings).
        DENSE: One prediction per spatial location (segmentation, depth).
        INSTANCES: A variable-length set of objects per sample — a box and a class for
            detection; masks and keypoints when those tasks land.
    """

    GLOBAL = "global"
    DENSE = "dense"
    INSTANCES = "instances"


class Stream(StrEnum):
    """Standard names of backbone feature streams.

    An open vocabulary: a multi-encoder backbone may produce streams under names of its own.
    Each member names a *shape class*, so a product of a different shape gets a different name.

    Attributes:
        FEATURES: ``[B, D]`` — the pooled per-sample vector; what GLOBAL heads read.
        ENCODER: ``[B, D, H', W']`` — the encoder's last spatial feature map.
        DECODER: ``[B, D, H, W]`` — the decoder's dense map; what dense kinds read.
        LOGITS: Task-shaped final outputs of a fused network, consumed through an identity head.
        EMBEDDINGS: ``[B, N, D]`` — aligned per-view embeddings; what contrastive criteria read.
    """

    FEATURES = "features"
    ENCODER = "encoder"
    DECODER = "decoder"
    LOGITS = "logits"
    EMBEDDINGS = "embeddings"


class Geometry(StrEnum):
    """How a value is transformed with the image during augmentation.

    Declared as a class-level fact by input loaders and target encoders, derived into the
    transform seam as the pipeline is built. ``BOXES`` also fixes the value's shape between ``load`` and
    ``encode``: ``(float32 [N, 4] xyxy-pixel array, list of class names)``. Measured on
    albumentationsx 2.3.7: oriented boxes and keypoints each have their own params there, so
    a future member is one entry here plus one in the seam.

    Attributes:
        NONE: Not in image space — labels, scalars.
        IMAGE: Light: interpolated smoothly, normalized.
        MASK: Per-pixel labels: nearest-neighbour geometry, never normalized.
        BOXES: Axis-aligned rectangles with their class names, in xyxy pixels.
    """

    NONE = "none"
    IMAGE = "image"
    MASK = "mask"
    BOXES = "boxes"


class Modality(StrEnum):
    """Standard names of model inputs, and the modalities they carry.

    Open, like ``Stream``: an experiment with an extra input names it freely, and a new
    modality is a new member here before it is anything else.

    Attributes:
        IMAGE: Pixel input, the default vision modality.
        EMBEDDING: Precomputed feature vectors instead of raw pixels.
        TEXT: Tokenized text (CLIP-style dual-encoder setups).
    """

    IMAGE = "image"
    EMBEDDING = "embedding"
    TEXT = "text"
