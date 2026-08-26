"""Domain vocabulary: the task axes ``OutputTopology`` x ``InputTopology`` x ``Objective`` x ``Modality``, and the data path's names."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Phase of a run a component is operating in."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class OutputTopology(StrEnum):
    """Output structure of a task — what one prediction looks like.

    A closed set: outer layers interpret each member by key, so a new member changes what the
    framework can express.

    Attributes:
        GLOBAL: One prediction vector per sample (classification, regression, embeddings).
        DENSE: One prediction per spatial location (segmentation, depth).
        INSTANCES: A variable-length set of objects per sample — a box and a class for
            detection; masks and keypoints when those tasks land.
    """

    GLOBAL = "global"
    DENSE = "dense"
    INSTANCES = "instances"


class InputTopology(StrEnum):
    """Input structure of a task — how many inputs feed one prediction, and how.

    ``SINGLE`` is the default wherever the axis is not named; only the paired kinds write it.

    Attributes:
        SINGLE: One input per sample; the ordinary case.
        MULTIVIEW: N views of each sample through one shared encoder (Siamese setups).
        MULTISTREAM: A separate encoder per input stream (CLIP-style dual encoders).
    """

    SINGLE = "single"
    MULTIVIEW = "multiview"
    MULTISTREAM = "multistream"


class Objective(StrEnum):
    """Label semantics of a task — how targets supervise the output.

    Attributes:
        MULTICLASS: Exactly one class per prediction.
        BINARY: A single yes/no probability per prediction.
        MULTILABEL: Independent per-class probabilities.
        CONTINUOUS: Real-valued targets (regression).
        METRIC: No explicit target values — supervision comes from pair or
            triplet structure, or from the in-batch diagonal (metric learning).
    """

    MULTICLASS = "multiclass"
    BINARY = "binary"
    MULTILABEL = "multilabel"
    CONTINUOUS = "continuous"
    METRIC = "metric"


class Stream(StrEnum):
    """Standard names of backbone feature streams.

    An open vocabulary: a multi-encoder backbone may produce streams under names of its own.
    Each member names a *shape class*, so a product of a different shape gets a different name.

    Attributes:
        FEATURES: ``[B, D]`` — the pooled per-sample vector; what GLOBAL heads read.
        ENCODER: ``[B, D, H', W']`` — the encoder's last spatial feature map.
        DECODER: ``[B, D, H, W]`` — the decoder's dense map; what DENSE topologies read.
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
    transform seam by assembly. ``BOXES`` also fixes the value's shape between ``load`` and
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
    """Standard names of model inputs — the input-side task axis.

    Open, like ``Stream``: an experiment with an extra input names it freely.

    Attributes:
        IMAGE: Pixel input, the default vision modality.
        EMBEDDING: Precomputed feature vectors instead of raw pixels.
        TEXT: Tokenized text (CLIP-style dual-encoder setups).
    """

    IMAGE = "image"
    EMBEDDING = "embedding"
    TEXT = "text"
