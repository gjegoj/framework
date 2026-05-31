"""Unit tests for the data layer on a synthetic, offline image dataset."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
import torch

from src.data import (
    CsvDataSource,
    DataModule,
    FloatCodec,
    LabelIndexCodec,
    MultiLabelBinarizeCodec,
    TargetBinding,
    build_basic_transform,
    collate_samples,
    split_dataframe,
)
from src.data.dataset import Dataset
from src.data.loaders import ImageLoader

LABELS = ["cat", "dog", "cow"]


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Create 15 synthetic 32x32 RGB jpgs across 3 classes and a CSV indexing them."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for index in range(15):
        array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        path = image_dir / f"{index}.jpg"
        cv2.imwrite(str(path), array)
        rows.append({"image_path": str(path), "label": LABELS[index % 3]})
    csv = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", codec=LabelIndexCodec())


class TestLabelIndexCodec:
    def test_fit_infers_sorted_classes(self) -> None:
        codec = LabelIndexCodec()
        codec.fit(["dog", "cat", "cow", "cat"])
        assert codec.num_classes == 3
        assert codec.class_mapping == {0: "cat", 1: "cow", 2: "dog"}

    def test_encode_returns_long_index(self) -> None:
        codec = LabelIndexCodec()
        codec.fit(["cat", "dog"])
        encoded = codec.encode("dog")
        assert encoded.item() == 1
        assert encoded.dtype.is_floating_point is False

    def test_unknown_label_raises(self) -> None:
        codec = LabelIndexCodec()
        codec.fit(["cat", "dog"])
        with pytest.raises(KeyError, match="Unknown label"):
            codec.encode("cow")

    def test_fixed_mapping_skips_fit(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "a", 1: "b"})
        codec.fit(["a", "a", "a"])  # no-op
        assert codec.num_classes == 2


class TestMultiLabelBinarizeCodec:
    def test_fit_builds_sorted_vocab(self) -> None:
        codec = MultiLabelBinarizeCodec()
        codec.fit(["cat,dog", "dog,cow", "cat"])
        assert codec.num_classes == 3
        assert codec.class_mapping == {0: "cat", 1: "cow", 2: "dog"}

    def test_encode_multihot(self) -> None:
        codec = MultiLabelBinarizeCodec()
        codec.fit(["cat,dog", "cow"])
        vec = codec.encode("cat,cow")
        assert vec.dtype == torch.float
        assert vec.shape == (3,)
        assert vec[codec.class_mapping[0] == "cat" and 0 or list(codec.class_mapping.values()).index("cat")].item() == 1.0

    def test_encode_multihot_correct_positions(self) -> None:
        codec = MultiLabelBinarizeCodec()
        codec.fit(["a,b,c"])
        # sorted vocab: a=0, b=1, c=2
        vec = codec.encode("a,c")
        assert vec.tolist() == [1.0, 0.0, 1.0]

    def test_separator_param(self) -> None:
        codec = MultiLabelBinarizeCodec(separator="|")
        codec.fit(["x|y", "z"])
        assert codec.num_classes == 3

    def test_unknown_label_raises(self) -> None:
        codec = MultiLabelBinarizeCodec()
        codec.fit(["cat,dog"])
        with pytest.raises(KeyError, match="Unknown label"):
            codec.encode("cat,fish")

    def test_fixed_mapping_skips_fit(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "a", 1: "b"})
        codec.fit(["a,b,c"])  # no-op — c would be a new class but mapping is fixed
        assert codec.num_classes == 2


class TestFloatCodec:
    def test_encode_scalar(self) -> None:
        codec = FloatCodec()
        codec.fit([1.0, 2.5, 3.0])
        t = codec.encode("2.5")
        assert t.dtype == torch.float
        assert t.ndim == 0
        assert t.item() == pytest.approx(2.5)

    def test_num_classes_is_none(self) -> None:
        codec = FloatCodec()
        codec.fit([1, 2, 3])
        assert codec.num_classes is None


class TestTaskCodecs:
    def test_binary_codec_shapes(self) -> None:
        from src.tasks.codecs import BinaryTaskCodec
        view = BinaryTaskCodec().adapt(torch.tensor([0, 1, 1, 0]))
        assert view.loss.shape == (4, 1) and view.loss.dtype == torch.float
        assert view.metric.shape == (4, 1) and view.metric.dtype == torch.long

    def test_multilabel_codec_shapes(self) -> None:
        from src.tasks.codecs import MultilabelTaskCodec
        target = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)
        view = MultilabelTaskCodec().adapt(target)
        assert view.loss.dtype == torch.float
        assert view.metric.dtype == torch.long

    def test_continuous_codec_shapes(self) -> None:
        from src.tasks.codecs import ContinuousTaskCodec
        view = ContinuousTaskCodec().adapt(torch.tensor([1.5, 2.3, 0.1]))
        assert view.loss.shape == (3, 1) and view.loss.dtype == torch.float
        assert view.metric.shape == (3, 1) and view.metric.dtype == torch.float


class TestSplit:
    def test_split_sizes_sum_and_are_disjoint(self) -> None:
        frame = pd.DataFrame({"x": range(15)})
        splits = split_dataframe(frame, {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}, seed=42)
        assert sum(len(part) for part in splits.values()) == 15
        assert len(splits[Stage.TRAIN]) == 9
        all_values = pd.concat(splits.values())["x"].tolist()
        assert sorted(all_values) == list(range(15))  # no row lost or duplicated


class TestDatasetAndCollate:
    def test_item_is_image_tensor_and_long_target(self, csv_path: Path) -> None:
        frame = pd.read_csv(csv_path)
        dataset = Dataset(
            frame=frame,
            image_column="image_path",
            bindings=[_binding_fitted(frame)],
            transform=build_basic_transform((16, 16), [0.5] * 3, [0.5] * 3),
            loader=ImageLoader(),
        )
        sample = dataset[0]
        assert sample.inputs["image"].shape == (3, 16, 16)
        assert sample.inputs["image"].dtype.is_floating_point
        assert sample.targets["label"].dtype.is_floating_point is False

    def test_collate_stacks_batch(self, csv_path: Path) -> None:
        frame = pd.read_csv(csv_path)
        dataset = Dataset(
            frame=frame,
            image_column="image_path",
            bindings=[_binding_fitted(frame)],
            transform=build_basic_transform((16, 16), [0.5] * 3, [0.5] * 3),
            loader=ImageLoader(),
        )
        batch = collate_samples([dataset[i] for i in range(4)])
        assert batch.inputs["image"].shape == (4, 3, 16, 16)
        assert batch.targets["label"].shape == (4,)


class TestDataModule:
    def test_setup_infers_num_classes_and_yields_batch(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {stage: build_basic_transform((16, 16), [0.5] * 3, [0.5] * 3) for stage in Stage}
        datamodule = DataModule(
            source=CsvDataSource(str(csv_path)),
            bindings=[_binding()],
            image_column="image_path",
            transforms=transforms,
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            runtime=runtime,
            batch_size=4,
            seed=0,
        )
        datamodule.setup()

        assert runtime.num_classes == {"label": 3}
        assert sum(runtime.dataset_sizes.values()) == 15

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.inputs["image"].shape[1:] == (3, 16, 16)
        assert batch.targets["label"].dtype.is_floating_point is False


def _binding_fitted(frame: pd.DataFrame) -> TargetBinding:
    binding = _binding()
    binding.codec.fit(frame["label"])
    return binding
