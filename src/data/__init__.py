"""Data layer: sources, codecs, transforms, dataset, datamodule and collation.

Interface-adapter layer: turns external data (CSV/JSON, image files) into the
core ``Sample``/``Batch`` entities. Depends on core, never the reverse.
"""

from src.data.bindings import TargetBinding
from src.data.codecs import FloatCodec, LabelIndexCodec, MultiLabelBinarizeCodec, TargetCodec, target_codecs
from src.data.collate import collate_samples
from src.data.datamodule import DataModule
from src.data.dataset import Dataset
from src.data.loaders import ImageLoader
from src.data.sources import CsvDataSource, DataSource, data_sources
from src.data.split import split_dataframe
from src.data.transforms import (
    AlbumentationsTransform,
    BasicTransform,
    Transform,
    build_albumentations_transform,
    build_basic_transform,
)

__all__ = [
    "AlbumentationsTransform",
    "BasicTransform",
    "CsvDataSource",
    "DataModule",
    "DataSource",
    "Dataset",
    "FloatCodec",
    "ImageLoader",
    "LabelIndexCodec",
    "MultiLabelBinarizeCodec",
    "TargetBinding",
    "TargetCodec",
    "Transform",
    "build_albumentations_transform",
    "build_basic_transform",
    "collate_samples",
    "data_sources",
    "split_dataframe",
    "target_codecs",
]
