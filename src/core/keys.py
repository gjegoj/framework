"""Canonical string keys for the cross-layer input/feature contract.

Centralising these avoids magic strings duplicated across the data and model
layers: the data layer writes ``Batch.inputs[IMAGE]``; a backbone reads it and
writes ``FeatureBundle.streams[POOLED]``; a head declares which stream it wants
via ``HeadSpec.feature_key``.

## Stream shape conventions

``POOLED`` and ``DECODER`` / ``ENCODER_LAST`` are the two canonical contracts.
Backbone-specific streams (``"p3"``, ``"tokens"``, ``"cls_token"``, ...) are
documented in each backbone's docstring — not declared here.

| Key            | Shape          | Who provides it            |
|----------------|----------------|----------------------------|
| ``POOLED``     | ``[B, D]``     | every backbone (mandatory) |
| ``DECODER``    | ``[B, D, H, W]``| encoder-decoder backbones |
| ``ENCODER_LAST``| ``[B, D, H, W]``| conv backbones (spatial)  |

## Discoverability

If you set an unknown ``feature_key`` in your task config, ``backbone.feature_dim``
raises a ``KeyError`` listing what the backbone actually exposes.  Each backbone's
docstring has a **Streams** section with the available keys and their shapes.
"""

# ------------------------------------------------------ Input modality keys
IMAGE = "image"
TEXT = "text"

# ------------------------------------------------------ Feature stream keys

# The only *mandatory* output contract: a per-sample vector [B, D].
# Every backbone that supports GlobalTopology must provide this stream.
POOLED = "pooled"

# Dense decoder output [B, D, H, W] — encoder-decoder architectures (smp, …).
# Used by DenseTopology as the default feature_key for segmentation heads.
DECODER = "decoder"

# Raw spatial output of the last encoder layer [B, D, H, W].
# Exposes the encoder features *before* the decoder, allowing a classification
# head (e.g. smp.ClassificationHead) to do its own spatial pooling.
ENCODER_LAST = "encoder_last"

# ----------------------------------------------------- Logging / metric keys
# Fixed tokens of the logged scalar keys. The full key is composed inline where
# it is logged — losses as ``{LOSS}/{stage}/{component}`` (``loss/val/total``),
# task metrics as ``{task}/{metric}/{stage}``. Centralised here so the token
# names can be changed in one place and referenced from YAML via ``${key:LOSS}``.
LOSS = "loss"  # namespace prefix for loss scalars
TOTAL = "total"  # the aggregate (weighted-sum) loss component
