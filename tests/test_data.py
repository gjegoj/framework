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
    CacheOptions,
    CsvDataSource,
    DataLoaderOptions,
    DataModule,
    EmbeddingLoader,
    JsonDataSource,
    LabelEncoder,
    MultiLabelEncoder,
    ScalarEncoder,
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
    return TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping=_LABEL_MAPPING))


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
        from src.data.loaders import infer_loader_key

        series = pd.Series(["emb/a.npy", "emb/b.npy"])
        assert infer_loader_key(series) == "embedding"

    def test_image_paths_still_infer_image(self) -> None:
        from src.data.loaders import infer_loader_key

        series = pd.Series(["img/a.jpg", "img/b.png"])
        assert infer_loader_key(series) == "image"


class TestIdentityTransform:
    def test_tensorizes_input_vector(self) -> None:
        from src.transforms.sample import IdentityTransform

        sample = Sample(inputs={"embedding": np.arange(4, dtype=np.float32)})
        result = IdentityTransform().apply(sample)
        assert isinstance(result.inputs["embedding"], torch.Tensor)
        assert result.inputs["embedding"].dtype == torch.float32
        assert result.inputs["embedding"].shape == (4,)

    def test_passes_targets_through_unchanged(self) -> None:
        from src.transforms.sample import IdentityTransform

        target = torch.tensor(2)
        sample = Sample(inputs={"embedding": np.zeros(4, dtype=np.float32)}, targets={"label": target})
        result = IdentityTransform().apply(sample)
        assert result.targets["label"] is target


class TestLabelEncoder:
    def test_class_mapping_sets_vocab(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        assert codec.num_classes == 3
        assert codec.class_mapping == {0: "cat", 1: "cow", 2: "dog"}

    def test_to_tensor_returns_long_index(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        tensor = codec.to_tensor(codec.load("dog"))
        assert tensor.item() == 1
        assert tensor.dtype.is_floating_point is False

    def test_load_returns_class_index(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        assert codec.load("dog") == 1  # encoding happens in load (a transform-ready int)

    def test_load_unknown_label_raises(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(KeyError, match="Unknown label"):
            codec.load("cow")

    def test_load_index_is_transform_ready_for_an_aug(self) -> None:
        """load() yields the int index, so a label-aware aug can bump it; to_tensor wraps it."""
        codec = LabelEncoder(class_mapping={0: "0", 1: "90", 2: "180", 3: "270"})
        bumped = (codec.load("0") + 3) % 4  # e.g. a rotation aug applying 270° CCW
        assert codec.to_tensor(bumped).item() == 3

    def test_fit_validates_known_labels(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        codec.fit(["cat", "dog", "cat"])  # all known — no error

    def test_fit_raises_on_unknown_label(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(ValueError, match="not in class_mapping"):
            codec.fit(["cat", "dog", "cow"])

    def test_fit_raises_without_class_mapping(self) -> None:
        codec = LabelEncoder()
        with pytest.raises(ValueError, match="class_mapping"):
            codec.fit(["cat", "dog"])


class TestMultiLabelEncoder:
    def test_class_mapping_sets_vocab(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        assert codec.num_classes == 3

    def test_to_tensor_multihot(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        vec = codec.to_tensor(codec.load("cat,cow"))
        assert vec.dtype == torch.float
        assert vec.shape == (3,)

    def test_load_builds_multihot_at_correct_positions(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "a", 1: "b", 2: "c"})
        assert codec.load("a,c").tolist() == [1.0, 0.0, 1.0]  # multi-hot built in load

    def test_separator_param(self) -> None:
        codec = MultiLabelEncoder(separator="|", class_mapping={0: "x", 1: "y", 2: "z"})
        assert codec.num_classes == 3

    def test_load_unknown_label_raises(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(KeyError, match="Unknown label"):
            codec.load("cat,fish")

    def test_fit_validates_known_labels(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        codec.fit(["cat,dog", "cow,dog", "cat"])  # all known — no error

    def test_fit_raises_on_unknown_label(self) -> None:
        codec = MultiLabelEncoder(class_mapping={0: "cat", 1: "dog"})
        with pytest.raises(ValueError, match="not in class_mapping"):
            codec.fit(["cat,dog", "cat,fish"])

    def test_fit_raises_without_class_mapping(self) -> None:
        codec = MultiLabelEncoder()
        with pytest.raises(ValueError, match="class_mapping"):
            codec.fit(["cat,dog"])


class TestScalarEncoder:
    def test_to_tensor_scalar(self) -> None:
        codec = ScalarEncoder()
        codec.fit([1.0, 2.5, 3.0])
        t = codec.to_tensor(codec.load("2.5"))
        assert t.dtype == torch.float
        assert t.ndim == 0
        assert t.item() == pytest.approx(2.5)

    def test_num_classes_is_none(self) -> None:
        codec = ScalarEncoder()
        codec.fit([1, 2, 3])
        assert codec.num_classes is None


class TestRotate90WithLabel:
    def test_image_and_label_share_the_same_k(self) -> None:
        """The sampled quarter-turn drives both the image and the label, so they always agree."""
        from src.transforms import Rotate90WithLabel

        transform = Rotate90WithLabel(label_key="rotation", p=1.0)
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)  # asymmetric → rotation observable
        for k in range(4):
            assert np.array_equal(transform.apply(image, k=k), np.rot90(image, k))
            assert transform.apply_to_label(0, k=k) == k

    def test_label_wraps_modulo_four(self) -> None:
        from src.transforms import Rotate90WithLabel

        transform = Rotate90WithLabel(label_key="rotation", p=1.0)
        assert transform.apply_to_label(2, k=3) == 1  # (2 + 3) % 4
        assert transform.apply_to_label(3, k=1) == 0  # wraps

    def test_compose_threads_rotation_label_as_int(self) -> None:
        from src.transforms import Rotate90WithLabel

        compose = A.Compose([Rotate90WithLabel(label_key="rotation", p=1.0)], seed=0)
        result = compose(image=np.zeros((4, 4, 3), np.uint8), rotation=0)
        assert isinstance(result["rotation"], int)
        assert result["rotation"] in {0, 1, 2, 3}

    def test_label_key_is_configurable(self) -> None:
        """The bound data key is a parameter, so the transform is not tied to a task named 'rotation'."""
        from src.transforms import Rotate90WithLabel

        compose = A.Compose([Rotate90WithLabel(label_key="orientation", p=1.0)], seed=0)
        result = compose(image=np.zeros((4, 4, 3), np.uint8), orientation=0)
        assert "orientation" in result
        assert result["orientation"] in {0, 1, 2, 3}

    def test_end_to_end_label_encoder_index_is_bumped_then_tensorized(self) -> None:
        """Encoder.load yields the class index, the aug bumps it, to_tensor wraps the result."""
        from src.transforms import Rotate90WithLabel

        encoder = LabelEncoder(class_mapping={0: "0", 1: "90", 2: "180", 3: "270"})
        transform = AlbumentationsTransform(A.Compose([Rotate90WithLabel(label_key="rotation", p=1.0)], seed=1))
        sample = Sample(inputs={"image": np.zeros((4, 4, 3), np.uint8)}, targets={"rotation": encoder.load("0")})
        index = transform.apply(sample).targets["rotation"]
        assert index in {0, 1, 2, 3}
        assert encoder.to_tensor(index).item() == index


class TestTargetAdapters:
    def test_binary_codec_shapes(self) -> None:
        from src.tasks.adapters import BinaryTargetAdapter

        view = BinaryTargetAdapter().adapt(torch.tensor([0, 1, 1, 0]))
        assert view.loss.shape == (4, 1) and view.loss.dtype == torch.float
        assert view.metric.shape == (4, 1) and view.metric.dtype == torch.long

    def test_multilabel_codec_shapes(self) -> None:
        from src.tasks.adapters import MultilabelTargetAdapter

        target = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)
        view = MultilabelTargetAdapter().adapt(target)
        assert view.loss.dtype == torch.float
        assert view.metric.dtype == torch.long

    def test_continuous_codec_shapes(self) -> None:
        from src.tasks.adapters import ContinuousTargetAdapter

        view = ContinuousTargetAdapter().adapt(torch.tensor([1.5, 2.3, 0.1]))
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


class TestDataLoaderKwargs:
    def _dm(self, csv_path: Path, **extra: object) -> DataModule:
        return DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            dataloader_options=DataLoaderOptions(extra_kwargs=dict(extra)),
        )

    def test_extra_kwargs_reach_dataloader(self, csv_path: Path) -> None:
        dm = self._dm(csv_path, timeout=5)
        dm.setup()
        assert dm.train_dataloader().timeout == 5

    def test_framework_keys_win_over_extras(self, csv_path: Path) -> None:
        # Defensive merge: even a framework-owned key in extras must not override the real value.
        dm = self._dm(csv_path, batch_size=999)
        dm.setup()
        assert dm.train_dataloader().batch_size == 4


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


class TestMaskEncoderAndDensePipeline:
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
        from src.data import MaskEncoder

        mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        path = tmp_path / "m.png"
        cv2.imwrite(str(path), mask)
        codec = MaskEncoder()
        assert codec.spatial is True
        arr = codec.load(str(path))
        assert arr.shape == (2, 2)
        tensor = codec.to_tensor(arr)
        assert tensor.dtype == torch.long

    def test_mask_codec_missing_file_raises(self) -> None:
        from src.data import MaskEncoder

        with pytest.raises(FileNotFoundError, match="Mask not found"):
            MaskEncoder().load("/no/such/mask.png")

    def test_dense_pipeline_aligns_image_and_mask(self, seg_csv: Path) -> None:
        from src.data import MaskEncoder

        frame = pd.read_csv(seg_csv)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[TargetBinding("mask", "mask_path", MaskEncoder())],
            transform=_make_transform((16, 16), spatial=["mask"]),
        )
        sample = dataset[0]
        assert sample.inputs["image"].shape == (3, 16, 16)  # image resized
        assert sample.targets["mask"].shape == (16, 16)  # mask resized in lockstep
        assert sample.targets["mask"].dtype == torch.long
        assert sample.targets["mask"].max().item() <= 2  # nearest preserved class indices

    def test_meta_records_input_and_target_source_paths(self, seg_csv: Path) -> None:
        from src.data import MaskEncoder

        frame = pd.read_csv(seg_csv)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[TargetBinding("mask", "mask_path", MaskEncoder())],
            transform=_make_transform((16, 16), spatial=["mask"]),
        )
        sample = dataset[2]
        assert sample.meta["index"] == 2
        assert sample.meta["input_sources"]["image"] == str(frame.iloc[2]["image_path"])  # image file path
        assert sample.meta["target_sources"]["mask"] == str(frame.iloc[2]["mask_path"])  # mask file path


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

    def test_collate_transposes_meta_sources(self) -> None:
        samples = [
            Sample(
                inputs={"image": torch.zeros(3, 4, 4)},
                targets={"mask": torch.zeros(4, 4, dtype=torch.long)},
                meta={
                    "index": i,
                    "input_sources": {"image": f"/img/{i}.jpg"},
                    "target_sources": {"mask": f"/msk/{i}.png"},
                },
            )
            for i in range(3)
        ]
        batch = collate_samples(samples)
        assert batch.meta["index"] == [0, 1, 2]  # scalar → per-sample list
        assert batch.meta["input_sources"] == {
            "image": ["/img/0.jpg", "/img/1.jpg", "/img/2.jpg"]
        }  # dict → dict-of-lists
        assert batch.meta["target_sources"] == {"mask": ["/msk/0.png", "/msk/1.png", "/msk/2.png"]}


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
    binding.encoder.fit(frame["label"])
    return binding


class TestArrayCache:
    def test_warm_then_get_hit_and_miss(self) -> None:
        from src.data.cache import ArrayCache

        store = {"a": np.zeros((2, 2), np.uint8), "b": np.ones((2, 2), np.uint8)}
        cache = ArrayCache(max_bytes=10**6)
        cache.warm(["a", "b"], store.__getitem__, workers=2)
        assert cache.get("a") is not None
        assert cache.get("missing") is None
        assert cache.nbytes == store["a"].nbytes + store["b"].nbytes

    def test_byte_cap_stops_filling(self) -> None:
        from src.data.cache import ArrayCache

        items = {str(i): np.zeros((100, 100), np.uint8) for i in range(10)}  # 10 KB each
        cache = ArrayCache(max_bytes=25_000)  # room for ~2 items
        cache.warm(list(items), items.__getitem__, workers=1)
        assert cache.nbytes <= 25_000
        assert 0 < sum(cache.get(k) is not None for k in items) < 10

    def test_bad_item_is_skipped_not_fatal(self) -> None:
        from src.data.cache import ArrayCache

        def load(key: str) -> np.ndarray:
            if key == "bad":
                raise FileNotFoundError(key)
            return np.zeros((2, 2), np.uint8)

        cache = ArrayCache(max_bytes=10**6)
        cache.warm(["ok", "bad"], load, workers=2)
        assert cache.get("ok") is not None
        assert cache.get("bad") is None

    def test_disabled_cache_caches_nothing(self) -> None:
        from src.data.cache import ArrayCache

        cache = ArrayCache(max_bytes=0)
        cache.warm(["a"], lambda _: np.zeros((2, 2), np.uint8), workers=1)
        assert cache.get("a") is None
        assert cache.nbytes == 0


class TestCachingDecorators:
    def test_caching_loader_hit_and_miss(self) -> None:
        from src.data.cache import ArrayCache, CachingLoader
        from src.data.loaders import ImageLoader

        cache = ArrayCache(max_bytes=10**6)
        cached = np.full((2, 2, 3), 7, np.uint8)
        cache.warm(["/hit.png"], lambda _: cached, workers=1)
        loader = CachingLoader(ImageLoader(), cache)
        assert loader.file_based is True
        assert np.array_equal(loader.load("/hit.png"), cached)  # served from cache
        with pytest.raises(FileNotFoundError):
            loader.load("/does-not-exist.png")  # miss falls through to the inner loader

    def test_caching_codec_delegates_and_serves(self) -> None:
        from src.data.cache import ArrayCache, CachingTargetEncoder
        from src.data.encoders import MaskEncoder

        cache = ArrayCache(max_bytes=10**6)
        cached = np.array([[1, 2], [3, 4]], np.uint8)
        cache.warm(["/m.png"], lambda _: cached, workers=1)
        codec = CachingTargetEncoder(MaskEncoder(class_mapping={0: "bg", 1: "fg"}), cache)
        assert codec.spatial is True
        assert codec.num_classes == 2  # delegated
        assert np.array_equal(codec.load("/m.png"), cached)  # served from cache
        assert codec.to_tensor(cached).dtype.is_floating_point is False  # delegated (long)


class TestDataModuleCache:
    def test_cache_warms_wraps_and_serves(self, csv_path: Path) -> None:
        from src.data.cache import CachingLoader

        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            cache_options=CacheOptions(max_bytes=10**9),
        )
        dm.setup()
        assert isinstance(dm._input_bindings[0].loader, CachingLoader)
        assert dm._cache is not None and dm._cache.nbytes > 0
        first_path = str(dm._datasets[Stage.TRAIN]._frame.iloc[0]["image_path"])
        assert dm._cache.get(first_path) is not None  # warmed (no root_path → key == path)
        sample = dm._datasets[Stage.TRAIN][0]  # end-to-end getitem still works
        assert "image" in sample.inputs

    def test_cache_disabled_leaves_plain_loader(self, csv_path: Path) -> None:
        from src.data.loaders import ImageLoader

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
        assert isinstance(dm._input_bindings[0].loader, ImageLoader)
        assert dm._cache is None
