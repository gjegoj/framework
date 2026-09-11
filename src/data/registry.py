"""The names a data declaration may write: what reads a cell, what joins a batch, what holds a split.

Every one of these is an extension point of the same shape as the others in this framework — register
a class of your own with ``@<name>_registry.register("name")`` and a declaration finds it, or reach it
by ``_target_`` without registering at all. A registry is a convenience, never a gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry
from src.data.base import Collator, DataModule, InputEncoder, Preprocessor, TableSource, TargetEncoder

if TYPE_CHECKING:
    from src.data.cache import Cache

input_encoder_registry: Registry[InputEncoder] = Registry("input encoder")
"""What `preprocessing.inputs.<name>` writes: how one input's raw cell becomes a tensor."""

target_encoder_registry: Registry[TargetEncoder] = Registry("target encoder")
"""What `tasks.<name>.target_encoder` writes, where a run overrides the one its kind reads with."""

collator_registry: Registry[Collator] = Registry("collator")
"""What `preprocessing.collator` writes: how prepared samples become one batch."""

table_source_registry: Registry[TableSource] = Registry("table source")
"""What a `data.source` format writes; each source declares the suffixes it reads."""

data_module_registry: Registry[DataModule] = Registry("data module")
"""What `data` writes: where rows come from, how they divide, and what a split serves."""

preprocessor_registry: Registry[Preprocessor] = Registry("preprocessor")
"""What `preprocessing` writes: the object that runs the encoders and the collator."""

cache_registry: Registry[Cache] = Registry("cache")
"""What `preprocessing.cache` writes, for a pipeline whose loading is worth doing once."""
