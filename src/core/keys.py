"""Canonical string keys for the cross-layer input/feature contract.

Centralizing these avoids magic strings duplicated across the data and model
layers: the data layer writes ``Batch.inputs[IMAGE]``; a backbone reads it and
writes ``FeatureBundle.streams[POOLED]``; a head declares which stream it wants.
"""

# Model input modalities — keys in ``Sample.inputs`` / ``Batch.inputs``.
IMAGE = "image"

# Backbone feature streams — keys in ``FeatureBundle.streams``.
POOLED = "pooled"  # global per-sample vector, shape [B, D]
DECODER = "decoder"  # dense per-pixel feature map, shape [B, D, H, W]
