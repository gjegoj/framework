"""The names a data declaration may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry
from src.data.base import Collator, DataModule, InputEncoder, Preprocessor, TableSource, TargetEncoder

if TYPE_CHECKING:
    from src.data.cache import Cache

input_encoder_registry = Registry[InputEncoder]("input encoder")
target_encoder_registry = Registry[TargetEncoder]("target encoder")
collator_registry = Registry[Collator]("collator")
table_source_registry = Registry[TableSource]("table source")
data_module_registry = Registry[DataModule]("data module")
preprocessor_registry = Registry[Preprocessor]("preprocessor")
cache_registry: Registry[Cache] = Registry("cache")
