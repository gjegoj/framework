"""The data capability: from annotation tables to model-ready batches."""

from __future__ import annotations

from src.data.cache import LoaderCache, RamCache, cached
from src.data.collate import collate_samples
from src.data.datamodules import SourceWithTransforms, TableDataModule, YoloDataModule
from src.data.dataset import TableDataset
from src.data.encoders import (
    BinnedTargetEncoder,
    BoxesTargetEncoder,
    GaussianBinsTargetEncoder,
    LabelTargetEncoder,
    LinearBinsTargetEncoder,
    MaskTargetEncoder,
    MultiLabelTargetEncoder,
    ScalarTargetEncoder,
    TargetEncoder,
)
from src.data.loaders import ImageLoader, InputLoader
from src.data.schema import ColumnRole, DataSchema, InputColumn, TargetColumn
from src.data.sources import (
    CsvSource,
    FileSource,
    InMemorySource,
    JsonLinesSource,
    JsonSource,
    LimitedSource,
    Table,
    TableSource,
)
from src.data.split import Splitter, group_split, random_split, stratified_split

__all__ = [
    "BinnedTargetEncoder",
    "BoxesTargetEncoder",
    "ColumnRole",
    "CsvSource",
    "DataSchema",
    "FileSource",
    "GaussianBinsTargetEncoder",
    "ImageLoader",
    "InMemorySource",
    "InputColumn",
    "InputLoader",
    "JsonLinesSource",
    "JsonSource",
    "LabelTargetEncoder",
    "LimitedSource",
    "LinearBinsTargetEncoder",
    "LoaderCache",
    "MaskTargetEncoder",
    "MultiLabelTargetEncoder",
    "RamCache",
    "ScalarTargetEncoder",
    "SourceWithTransforms",
    "Splitter",
    "Table",
    "TableDataModule",
    "TableDataset",
    "TableSource",
    "TargetColumn",
    "TargetEncoder",
    "YoloDataModule",
    "cached",
    "collate_samples",
    "group_split",
    "random_split",
    "stratified_split",
]
