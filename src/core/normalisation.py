"""ImageNet's per-channel statistics, which every shipped default normalises by."""

from __future__ import annotations

from typing import Final

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
"""Per-channel mean of ImageNet, which every pretrained backbone here was fitted on.

Named once because two readers have to agree: the transforms that normalise, and the
samples grid that undoes the normalisation to draw the pixels back. Different numbers
on the two sides make a picture that looks like a model problem.

A default, not an assumption — a run that normalises differently says so at the root,
and both readers follow it through ``${mean}`` and ``${std}``.
"""

IMAGENET_STD: Final = (0.229, 0.224, 0.225)
"""The matching per-channel standard deviation."""
