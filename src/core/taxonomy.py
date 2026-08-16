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

    A closed set, like every axis here: outer layers interpret each member by key, so
    a new member is a change to what the framework can express, not a new string.

    Attributes:
        GLOBAL: One prediction vector per sample (classification, regression,
            and metric learning's embedding).
        DENSE: One prediction per spatial location (segmentation, depth).
        INSTANCES: A variable-length set of objects per sample, each carrying
            task-specific instance-level attributes — a box and a class for
            detection; masks and keypoints when those tasks land.
    """

    GLOBAL = "global"
    DENSE = "dense"
    INSTANCES = "instances"


class InputTopology(StrEnum):
    """Input structure of a task — how many inputs feed one prediction, and how.

    ``SINGLE`` is the default everywhere an axis is not named: presets and
    config both assume it, so only the paired kinds ever write this axis down.

    Attributes:
        SINGLE: One input per sample; the ordinary case.
        MULTIVIEW: N views of each sample through one shared encoder
            (Siamese setups; supervision compares the views).
        MULTISTREAM: A separate encoder per input stream (CLIP-style dual
            encoders; supervision aligns the streams).
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

    An open vocabulary, unlike the axes: these members name the conventional cases,
    and a multi-encoder backbone may produce streams under names of its own.

    Each member names a *shape class* — the name is a contract about what the tensor
    looks like, which is what heads and topologies rely on. A product of a different
    shape gets a different name even when it is built from another stream (DECODER is
    made from ENCODER features; EMBEDDINGS stacks per-view FEATURES).

    Attributes:
        FEATURES: ``[B, D]`` — the pooled per-sample vector; the single
            stream of simple backbones and what GLOBAL heads read.
        ENCODER: ``[B, D, H', W']`` — the encoder's last spatial feature map
            (encoder-decoder backbones).
        DECODER: ``[B, D, H, W]`` — the decoder's dense map; what DENSE
            topologies read.
        LOGITS: Task-shaped final outputs of a fused network, consumed
            through an identity head.
        EMBEDDINGS: ``[B, N, D]`` — aligned per-view embeddings (multi-encoder
            now, multi-view later); the carrier contrastive criteria consume.
    """

    FEATURES = "features"
    ENCODER = "encoder"
    DECODER = "decoder"
    LOGITS = "logits"
    EMBEDDINGS = "embeddings"


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
