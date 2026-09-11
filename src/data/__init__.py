"""Data preparation: encoders, the preprocessor that runs them, the cache they fill, and the table module.

Importing this package is what makes its names resolvable: every implementation below registers as its
module runs, so `preprocessing.inputs.image={name: image}` in a config finds its encoder here.
"""

from __future__ import annotations

from src.data.base import (
    Collator,
    DataModule,
    Encoder,
    InputEncoder,
    Preprocessor,
    TableSource,
    TargetEncoder,
)
from src.data.cache import Cache, RamCache
from src.data.collate import StackCollator
from src.data.encoders import (
    GaussianBinsEncoder,
    ImageEncoder,
    LabelEncoder,
    LinearBinsEncoder,
    MaskEncoder,
    MultilabelEncoder,
    ScalarEncoder,
)
from src.data.preprocessor import StandardPreprocessor
from src.data.sources import FileSource
from src.data.table import TableDataModule

__all__ = [
    "Cache",
    "Collator",
    "DataModule",
    "Encoder",
    "FileSource",
    "GaussianBinsEncoder",
    "ImageEncoder",
    "InputEncoder",
    "LabelEncoder",
    "LinearBinsEncoder",
    "MaskEncoder",
    "MultilabelEncoder",
    "Preprocessor",
    "RamCache",
    "ScalarEncoder",
    "StackCollator",
    "StandardPreprocessor",
    "TableDataModule",
    "TableSource",
    "TargetEncoder",
]
