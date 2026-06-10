"""Data layer: sources, codecs, transforms, dataset, datamodule and collation.

Interface-adapter layer: turns external data (CSV/JSON, image files) into the
core ``Sample``/``Batch`` entities. Depends on core, never the reverse.
"""

from src.data.bindings import InputBinding, TargetBinding
from src.data.codecs import (
    FloatCodec,
    LabelIndexCodec,
    MaskCodec,
    MultiLabelBinarizeCodec,
    TargetCodec,
    target_codecs,
)
from src.data.collate import collate_samples
from src.data.datamodule import DataModule
from src.data.dataset import Dataset
from src.data.loaders import ImageLoader, InputLoader, TextLoader, input_loaders
from src.data.sources import CsvDataSource, DataSource, FileDataSource, JsonDataSource, data_sources
from src.data.split import split_dataframe
from src.transforms.input import AlbumentationsTransform, Transform

__all__ = [
    "AlbumentationsTransform",
    "CsvDataSource",
    "DataModule",
    "DataSource",
    "Dataset",
    "FileDataSource",
    "FloatCodec",
    "ImageLoader",
    "InputBinding",
    "InputLoader",
    "TextLoader",
    "input_loaders",
    "JsonDataSource",
    "LabelIndexCodec",
    "MaskCodec",
    "MultiLabelBinarizeCodec",
    "TargetBinding",
    "TargetCodec",
    "Transform",
    "collate_samples",
    "data_sources",
    "split_dataframe",
    "target_codecs",
]
