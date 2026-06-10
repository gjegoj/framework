"""Registry of batch transforms selectable from YAML by name.

A batch transform mixes whole samples in a collated ``Batch`` (MixUp, CutMix,
Mosaic). Register a custom one with ``@batch_transforms.register("key")`` or
bypass the registry with a ``_target_`` spec.
"""

from __future__ import annotations

from src.core.ports import BatchTransform
from src.core.registry import Registry

batch_transforms: Registry[BatchTransform] = Registry("batch_transform")
