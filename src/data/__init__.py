"""The data capability: from annotation tables to model-ready batches."""

from __future__ import annotations

from src.data.cache import LoaderCache, RamCache, cached
from src.data.collate import collate_samples
from src.data.datamodules import DataModule, DeclaredSource, TableDataModule
from src.data.dataset import TableDataset
from src.data.encoders import (
    BinnedTargetEncoder,
    BoxesTargetEncoder,
    FileTargetEncoder,
    GaussianBinsTargetEncoder,
    LabelTargetEncoder,
    LinearBinsTargetEncoder,
    MaskTargetEncoder,
    MultiLabelTargetEncoder,
    ScalarTargetEncoder,
    TargetEncoder,
    VocabularyTargetEncoder,
)
from src.data.loaders import ImageLoader, InputLoader, single_threaded_cv2
from src.data.schema import ColumnRole, DataSchema, InputColumn, TargetColumn
from src.data.sources import (
    CsvSource,
    FileSource,
    JsonLinesSource,
    JsonSource,
    Table,
    TableSource,
)
from src.data.split import Splitter, group_split, random_split, stratified_split

__all__ = [
    "BinnedTargetEncoder",
    "BoxesTargetEncoder",
    "ColumnRole",
    "CsvSource",
    "DataModule",
    "DataSchema",
    "DeclaredSource",
    "FileSource",
    "FileTargetEncoder",
    "GaussianBinsTargetEncoder",
    "ImageLoader",
    "InputColumn",
    "InputLoader",
    "JsonLinesSource",
    "JsonSource",
    "LabelTargetEncoder",
    "LinearBinsTargetEncoder",
    "LoaderCache",
    "MaskTargetEncoder",
    "MultiLabelTargetEncoder",
    "RamCache",
    "ScalarTargetEncoder",
    "Splitter",
    "Table",
    "TableDataModule",
    "TableDataset",
    "TableSource",
    "TargetColumn",
    "TargetEncoder",
    "VocabularyTargetEncoder",
    "cached",
    "collate_samples",
    "group_split",
    "random_split",
    "single_threaded_cv2",
    "stratified_split",
]
