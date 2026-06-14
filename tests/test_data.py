"""Unit tests for the data layer on a synthetic, offline image dataset."""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.core.entities import Sample
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    AlbumentationsTransform,
    CsvDataSource,
    DataModule,
    EmbeddingLoader,
    FloatCodec,
    JsonDataSource,
    LabelIndexCodec,
    MultiLabelBinarizeCodec,
    TargetBinding,
    collate_samples,
    split_dataframe,
)
from src.data.datamodule import _build_input_bindings
from src.data.dataset import Dataset


def _make_transform(
    size: tuple[int, int] = (16, 16),
    spatial: list[str] | None = None,
) -> AlbumentationsTransform:
    h, w = size
    compose = A.Compose([A.Resize(h, w, mask_interpolation=0), A.Normalize(), ToTensorV2()])
    return AlbumentationsTransform(compose, spatial_targets=spatial)


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


_LABEL_MAPPING: dict[int, str] = {0: "cat", 1: "cow", 2: "dog"}
_TAGS_MAPPING: dict[int, str] = {0: "indoor", 1: "large", 2: "outdoor", 3: "small"}


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", codec=LabelIndexCodec(class_mapping=_LABEL_MAPPING))


class TestEmbeddingLoader:
    def test_load_reads_vector_as_float32(self, tmp_path: Path) -> None:
        path = tmp_path / "vec.npy"
        np.save(path, np.arange(8, dtype=np.float64))
        vector = EmbeddingLoader().load(str(path))
        assert vector.shape == (8,)
        assert vector.dtype == np.float32

    def test_load_non_1d_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "matrix.npy"
        np.save(path, np.zeros((4, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="1-D"):
            EmbeddingLoader().load(str(path))

    def test_is_file_based(self) -> None:
        assert EmbeddingLoader().file_based is True


class TestInferLoaderKey:
    def test_npy_paths_infer_embedding(self) -> None:
        from src.data.loaders import _infer_loader_key

        series = pd.Series(["emb/a.npy", "emb/b.npy"])
        assert _infer_loader_key(series) == "embedding"

    def test_image_paths_still_infer_image(self) -> None:
        from src.data.loaders import _infer_loader_key

        series = pd.Series(["img/a.jpg", "img/b.png"])
        assert _infer_loader_key(series) == "image"


class TestIdentityTransform:
    def test_tensorizes_input_vector(self) -> None:
        from src.transforms.input import IdentityTransform

        sample = Sample(inputs={"embedding": np.arange(4, dtype=np.float32)})
        result = IdentityTransform().apply(sample)
        assert isinstance(result.inputs["embedding"], torch.Tensor)
        assert result.inputs["embedding"].dtype == torch.float32
        assert result.inputs["embedding"].shape == (4,)

    def test_passes_targets_through_unchanged(self) -> None:
        from src.transforms.input import IdentityTransform

        target = torch.tensor(2)
        sample = Sample(inputs={"embedding": np.zeros(4, dtype=np.float32)}, targets={"label": target})
        result = IdentityTransform().apply(sample)
        assert result.targets["label"] is target


class TestLabelIndexCodec:
    def test_class_mapping_sets_vocab(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        assert codec.num_classes == 3
        assert codec.class_mapping == {0: "cat", 1: "cow", 2: "dog"}

    def test_to_tensor_returns_long_index(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "dog"})
        tensor = codec.to_tensor(codec.load("dog"))
        assert tensor.item() == 1
        assert tensor.dtype.is_floating_point is False

    def test_load_is_identity(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "dog"})
        assert codec.load("cat") == "cat"

    def test_to_tensor_unknown_label_raises(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(KeyError, match="Unknown label"):
            codec.to_tensor("cow")

    def test_fit_validates_known_labels(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "dog"})
        codec.fit(["cat", "dog", "cat"])  # all known — no error

    def test_fit_raises_on_unknown_label(self) -> None:
        codec = LabelIndexCodec(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(ValueError, match="not in class_mapping"):
            codec.fit(["cat", "dog", "cow"])

    def test_fit_raises_without_class_mapping(self) -> None:
        codec = LabelIndexCodec()
        with pytest.raises(ValueError, match="class_mapping"):
            codec.fit(["cat", "dog"])


class TestMultiLabelBinarizeCodec:
    def test_class_mapping_sets_vocab(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        assert codec.num_classes == 3

    def test_to_tensor_multihot(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        vec = codec.to_tensor(codec.load("cat,cow"))
        assert vec.dtype == torch.float
        assert vec.shape == (3,)

    def test_to_tensor_correct_positions(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "a", 1: "b", 2: "c"})
        vec = codec.to_tensor("a,c")
        assert vec.tolist() == [1.0, 0.0, 1.0]

    def test_separator_param(self) -> None:
        codec = MultiLabelBinarizeCodec(separator="|", class_mapping={0: "x", 1: "y", 2: "z"})
        assert codec.num_classes == 3

    def test_to_tensor_unknown_label_raises(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(KeyError, match="Unknown label"):
            codec.to_tensor("cat,fish")

    def test_fit_validates_known_labels(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        codec.fit(["cat,dog", "cow,dog", "cat"])  # all known — no error

    def test_fit_raises_on_unknown_label(self) -> None:
        codec = MultiLabelBinarizeCodec(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(ValueError, match="not in class_mapping"):
            codec.fit(["cat,dog", "cat,fish"])

    def test_fit_raises_without_class_mapping(self) -> None:
        codec = MultiLabelBinarizeCodec()
        with pytest.raises(ValueError, match="class_mapping"):
            codec.fit(["cat,dog"])


class TestFloatCodec:
    def test_to_tensor_scalar(self) -> None:
        codec = FloatCodec()
        codec.fit([1.0, 2.5, 3.0])
        t = codec.to_tensor(codec.load("2.5"))
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


class TestMaxSamples:
    def test_int_caps_rows(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            max_samples=6,
        )
        dm.setup()
        total = sum(len(dm._datasets[s]) for s in dm._datasets)
        assert total == 6

    def test_float_caps_fraction(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            max_samples=0.5,
        )
        dm.setup()
        total = sum(len(dm._datasets[s]) for s in dm._datasets)
        assert total == 8  # 50% of 15 rows = 7.5 → 8 (pandas rounds up)

    def test_none_keeps_all_rows(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
        )
        dm.setup()
        assert sum(len(dm._datasets[s]) for s in dm._datasets) == 15


class TestSplitDataframe:
    def _frame(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        labels = (["cat"] * 20 + ["dog"] * 20 + ["cow"] * 20)[:n]
        return pd.DataFrame({"label": labels, "score": rng.random(n)})

    def test_random_split_sizes(self) -> None:
        frame = self._frame()
        parts = split_dataframe(frame, {Stage.TRAIN: 0.7, Stage.VAL: 0.15, Stage.TEST: 0.15}, seed=0)
        assert sum(len(p) for p in parts.values()) == 60

    def test_categorical_stratify_preserves_distribution(self) -> None:
        frame = self._frame()
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.15, Stage.TEST: 0.15},
            seed=0,
            stratify_column="label",
        )
        for part in parts.values():
            counts = part["label"].value_counts(normalize=True)
            for cls in ["cat", "dog", "cow"]:
                assert abs(counts.get(cls, 0) - 1 / 3) < 0.15

    def test_numeric_stratify_works(self) -> None:
        frame = self._frame()
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.3},
            seed=0,
            stratify_column="score",
        )
        assert sum(len(p) for p in parts.values()) == 60

    def test_multilabel_stratify_works(self) -> None:
        rng = np.random.default_rng(1)
        labels = [
            ",".join(rng.choice(["a", "b", "c"], size=rng.integers(1, 3), replace=False).tolist()) for _ in range(60)
        ]
        frame = pd.DataFrame({"tags": labels})
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.3},
            seed=0,
            stratify_column="tags",
        )
        assert sum(len(p) for p in parts.values()) == 60

    def test_missing_stratify_column_raises(self) -> None:
        frame = self._frame()
        with pytest.raises(ValueError, match="stratify_column"):
            split_dataframe(frame, {Stage.TRAIN: 0.8, Stage.VAL: 0.2}, seed=0, stratify_column="nonexistent")


class TestDataSources:
    def test_csv_reads_and_concatenates(self, tmp_path: Path) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
        pd.DataFrame({"x": [3]}).to_csv(b, index=False)
        frame = CsvDataSource([str(a), str(b)]).read()
        assert frame["x"].tolist() == [1, 2, 3]

    def test_json_reads_array_of_records(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('[{"image_path": "a.jpg", "label": "cat"}, {"image_path": "b.jpg", "label": "dog"}]')
        frame = JsonDataSource(str(path)).read()
        assert frame["label"].tolist() == ["cat", "dog"]

    def test_json_concatenates_multiple_files(self, tmp_path: Path) -> None:
        p1 = tmp_path / "1.json"
        p2 = tmp_path / "2.json"
        p1.write_text('[{"x": 1}]')
        p2.write_text('[{"x": 2}, {"x": 3}]')
        frame = JsonDataSource([str(p1), str(p2)]).read()
        assert frame["x"].tolist() == [1, 2, 3]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Data source file not found"):
            CsvDataSource(str(tmp_path / "nope.csv")).read()

    def test_empty_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one path"):
            JsonDataSource([])


class TestMaskCodecAndDensePipeline:
    @pytest.fixture
    def seg_csv(self, tmp_path: Path) -> Path:
        """5 images + matching index-mask ONGs (classes 0..2)."""
        img_dir = tmp_path / "img"
        msk_dir = tmp_path / "msk"
        img_dir.mkdir()
        msk_dir.mkdir()
        rng = np.random.default_rng(0)
        rows = []
        for i in range(5):
            cv2.imwrite(str(img_dir / f"{i}.jpg"), rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))
            mask = rng.integers(0, 3, (40, 40), dtype=np.uint8)  # class indices 0..2
            cv2.imwrite(str(msk_dir / f"{i}.png"), mask)
            rows.append({"image_path": str(img_dir / f"{i}.jpg"), "mask_path": str(msk_dir / f"{i}.png")})
        csv = tmp_path / "seg.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_mask_codec_loads_and_finalizes(self, tmp_path: Path) -> None:
        from src.data import MaskCodec

        mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        path = tmp_path / "m.png"
        cv2.imwrite(str(path), mask)
        codec = MaskCodec()
        assert codec.spatial is True
        arr = codec.load(str(path))
        assert arr.shape == (2, 2)
        tensor = codec.to_tensor(arr)
        assert tensor.dtype == torch.long

    def test_mask_codec_missing_file_raises(self) -> None:
        from src.data import MaskCodec

        with pytest.raises(FileNotFoundError, match="Mask not found"):
            MaskCodec().load("/no/such/mask.png")

    def test_dense_pipeline_aligns_image_and_mask(self, seg_csv: Path) -> None:
        from src.data import MaskCodec

        frame = pd.read_csv(seg_csv)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[TargetBinding("mask", "mask_path", MaskCodec())],
            transform=_make_transform((16, 16), spatial=["mask"]),
        )
        sample = dataset[0]
        assert sample.inputs["image"].shape == (3, 16, 16)  # image resized
        assert sample.targets["mask"].shape == (16, 16)  # mask resized in lockstep
        assert sample.targets["mask"].dtype == torch.long
        assert sample.targets["mask"].max().item() <= 2  # nearest preserved class indices


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
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[_binding_fitted(frame)],
            transform=_make_transform((16, 16)),
        )
        sample = dataset[0]
        assert sample.inputs["image"].shape == (3, 16, 16)
        assert sample.inputs["image"].dtype.is_floating_point
        assert sample.targets["label"].dtype.is_floating_point is False

    def test_collate_stacks_batch(self, csv_path: Path) -> None:
        frame = pd.read_csv(csv_path)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[_binding_fitted(frame)],
            transform=_make_transform((16, 16)),
        )
        batch = collate_samples([dataset[i] for i in range(4)])
        assert batch.inputs["image"].shape == (4, 3, 16, 16)
        assert batch.targets["label"].shape == (4,)


class TestDataModule:
    def test_setup_infers_num_classes_and_yields_batch(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {stage: _make_transform((16, 16)) for stage in Stage}
        datamodule = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
        )
        datamodule.setup()

        assert runtime.num_classes == {"label": 3}
        assert sum(len(datamodule._datasets[s]) for s in datamodule._datasets) == 15

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.inputs["image"].shape[1:] == (3, 16, 16)
        assert batch.targets["label"].dtype.is_floating_point is False

    def test_presplit_mode_fits_on_train_only(self, csv_path: Path, tmp_path: Path) -> None:
        """Pre-split: separate train/val CSVs; codec fitted on train, applied to val."""

        full_frame = pd.read_csv(csv_path)
        train_csv = tmp_path / "train.csv"
        val_csv = tmp_path / "val.csv"
        full_frame.iloc[:10].to_csv(train_csv, index=False)
        full_frame.iloc[10:].to_csv(val_csv, index=False)

        runtime = RuntimeContext()
        transforms = {s: _make_transform((16, 16)) for s in Stage}
        datamodule = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            staged_sources={
                Stage.TRAIN: CsvDataSource(str(train_csv)),
                Stage.VAL: CsvDataSource(str(val_csv)),
            },
        )
        datamodule.setup()

        assert runtime.num_classes == {"label": 3}
        assert len(datamodule._datasets[Stage.TRAIN]) == 10
        assert len(datamodule._datasets[Stage.VAL]) == 5

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.inputs["image"].shape[1:] == (3, 16, 16)


def _binding_fitted(frame: pd.DataFrame) -> TargetBinding:
    binding = _binding()
    binding.codec.fit(frame["label"])
    return binding
