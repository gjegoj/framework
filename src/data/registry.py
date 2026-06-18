"""The data layer's registries: target encoders, input loaders, data sources.

Co-located here (one home, like ``models``/``metrics``) instead of inside each content file.
The ABCs they're typed against live in the data layer, so they are imported under
``TYPE_CHECKING`` only — at runtime this module just creates empty registries, which the
content modules import and register against (no import cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader
    from src.data.sources import DataSource

target_encoders: Registry[TargetEncoder] = Registry("target_encoder")
input_loaders: Registry[InputLoader] = Registry("input_loader")
data_sources: Registry[DataSource] = Registry("data_source")
