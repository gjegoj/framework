"""Data layer: sources, encoders, transforms, dataset, datamodule and collation.

Interface-adapter layer: turns external data (CSV/JSON, image files) into the
core ``Sample``/``Batch`` entities. Depends on core, never the reverse.
"""

from src.data.bindings import InputBinding, TargetBinding
from src.data.collate import collate_samples
from src.data.datamodule import DataModule
from src.data.dataset import Dataset
from src.data.encoders import (
    LabelEncoder,
    MaskEncoder,
    MultiLabelEncoder,
    ScalarEncoder,
    TargetEncoder,
)
from src.data.loaders import EmbeddingLoader, ImageLoader, InputLoader, TextLoader
from src.data.registry import data_sources, input_loaders, target_encoders
from src.data.sources import CsvDataSource, DataSource, FileDataSource, JsonDataSource
from src.data.split import split_dataframe
from src.transforms.sample import AlbumentationsTransform, Transform

__all__ = [
    "AlbumentationsTransform",
    "CsvDataSource",
    "DataModule",
    "DataSource",
    "Dataset",
    "EmbeddingLoader",
    "FileDataSource",
    "ScalarEncoder",
    "ImageLoader",
    "InputBinding",
    "InputLoader",
    "TextLoader",
    "input_loaders",
    "JsonDataSource",
    "LabelEncoder",
    "MaskEncoder",
    "MultiLabelEncoder",
    "TargetBinding",
    "TargetEncoder",
    "Transform",
    "collate_samples",
    "data_sources",
    "split_dataframe",
    "target_encoders",
]
