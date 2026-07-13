"""Target encoders: label/multilabel/scalar/null and the mask (dense) pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    LabelEncoder,
    MultiLabelEncoder,
    ScalarEncoder,
    TargetBinding,
)
from src.data.datamodule import _build_input_bindings
from src.data.dataset import Dataset
from tests.support.builders import make_transform as _make_transform

LABELS = ["cat", "dog", "cow"]


@pytest.fixture
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 RGB jpgs across 3 classes and a CSV indexing them."""
    return make_image_csv(count=15, size=32, seed=0, labels=LABELS)


class TestLabelEncoder:
    def test_class_mapping_sets_vocab(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"})
        assert codec.num_classes == 3
        assert codec.class_mapping == {0: "cat", 1: "cow", 2: "dog"}

    def test_is_neither_file_based_nor_spatial(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        assert codec.file_based is False  # raw column value — no root_path / source / cache
        assert codec.spatial is False  # not a mask — does not enter the geometric transform

    def test_to_tensor_returns_long_index(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        tensor = codec.to_tensor(codec.load("dog"))
        assert tensor.item() == 1
        assert tensor.dtype.is_floating_point is False

    def test_load_returns_class_index(self) -> None:
        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        assert codec.load("dog") == 1  # encoding happens in load (a transform-ready int)

    def test_load_index_is_transform_ready_for_an_aug(self) -> None:
        """load() yields the int index, so a label-aware aug can bump it; to_tensor wraps it."""
        codec = LabelEncoder(class_mapping={0: "0", 1: "90", 2: "180", 3: "270"})
        bumped = (codec.load("0") + 3) % 4  # e.g. a rotation aug applying 270° CCW
        assert codec.to_tensor(bumped).item() == 3


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


class TestCategoricalEncoderContract:
    """Behaviours shared verbatim by the categorical encoders — same contract, different value format."""

    @pytest.mark.parametrize(
        ("codec", "bad_value"),
        [
            pytest.param(LabelEncoder(class_mapping={0: "cat", 1: "dog"}), "cow", id="label"),
            pytest.param(MultiLabelEncoder(class_mapping={0: "cat", 1: "dog"}), "cat,fish", id="multilabel"),
        ],
    )
    def test_load_unknown_label_raises(self, codec: LabelEncoder | MultiLabelEncoder, bad_value: str) -> None:
        with pytest.raises(KeyError, match="Unknown label"):
            codec.load(bad_value)

    @pytest.mark.parametrize(
        ("codec", "known_values"),
        [
            pytest.param(LabelEncoder(class_mapping={0: "cat", 1: "dog"}), ["cat", "dog", "cat"], id="label"),
            pytest.param(
                MultiLabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"}),
                ["cat,dog", "cow,dog", "cat"],
                id="multilabel",
            ),
        ],
    )
    def test_fit_validates_known_labels(self, codec: LabelEncoder | MultiLabelEncoder, known_values: list[str]) -> None:
        codec.fit(known_values)  # all known — no error

    @pytest.mark.parametrize(
        ("codec", "values_with_unknown"),
        [
            pytest.param(LabelEncoder(class_mapping={0: "cat", 1: "dog"}), ["cat", "dog", "cow"], id="label"),
            pytest.param(
                MultiLabelEncoder(class_mapping={0: "cat", 1: "dog"}), ["cat,dog", "cat,fish"], id="multilabel"
            ),
        ],
    )
    def test_fit_raises_on_unknown_label(
        self, codec: LabelEncoder | MultiLabelEncoder, values_with_unknown: list[str]
    ) -> None:
        with pytest.raises(ValueError, match="not in class_mapping"):
            codec.fit(values_with_unknown)

    @pytest.mark.parametrize(
        ("codec", "values"),
        [
            pytest.param(LabelEncoder(), ["cat", "dog"], id="label"),
            pytest.param(MultiLabelEncoder(), ["cat,dog"], id="multilabel"),
        ],
    )
    def test_fit_raises_without_class_mapping(self, codec: LabelEncoder | MultiLabelEncoder, values: list[str]) -> None:
        with pytest.raises(ValueError, match="class_mapping"):
            codec.fit(values)


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


class TestSupportsSummary:
    """Capability Protocol: encoders that can summarize declare it; others stay silent."""

    def test_label_encoder_satisfies_protocol(self) -> None:
        from src.data.statistics import SupportsSummary

        codec = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        assert isinstance(codec, SupportsSummary)

    def test_scalar_encoder_satisfies_protocol(self) -> None:
        from src.data.statistics import SupportsSummary

        assert isinstance(ScalarEncoder(), SupportsSummary)

    def test_bare_target_encoder_stub_does_not_satisfy_protocol(self) -> None:
        from src.data.encoders import TargetEncoder
        from src.data.statistics import SupportsSummary

        class _StubEncoder(TargetEncoder):
            def fit(self, values: Iterable[Any]) -> None:
                pass

            def load(self, value: Any) -> Any:
                return value

            def to_tensor(self, value: Any) -> torch.Tensor:
                return torch.tensor(0)

        assert not isinstance(_StubEncoder(), SupportsSummary)

    def test_mask_encoder_does_not_satisfy_protocol(self) -> None:
        from src.data import MaskEncoder
        from src.data.statistics import SupportsSummary

        assert not isinstance(MaskEncoder(), SupportsSummary)

    def test_caching_encoder_over_summarizing_inner_satisfies_protocol(self) -> None:
        from src.data.cache import ArrayCache, caching_target_encoder
        from src.data.statistics import CategoricalDistribution, SupportsSummary

        cache = ArrayCache(max_bytes=10**6)
        inner = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        inner.fit(["cat", "dog"])
        cached = caching_target_encoder(inner, cache)
        assert isinstance(cached, SupportsSummary)
        # Delegation must produce the same distribution as the inner encoder directly.
        values = ["cat", "dog", "cat"]
        result = cached.summarize(values)
        assert isinstance(result, CategoricalDistribution)
        assert result.counts == {"cat": 2, "dog": 1}

    def test_caching_encoder_over_mask_encoder_does_not_satisfy_protocol(self) -> None:
        from src.data import MaskEncoder
        from src.data.cache import ArrayCache, caching_target_encoder
        from src.data.statistics import SupportsSummary

        cache = ArrayCache(max_bytes=10**6)
        cached = caching_target_encoder(MaskEncoder(class_mapping={0: "bg", 1: "fg"}), cache)
        assert not isinstance(cached, SupportsSummary)


class TestNullTargetEncoder:
    def test_load_and_to_tensor_produce_scalar_ignoring_value(self) -> None:
        from src.data.encoders import NullTargetEncoder

        encoder = NullTargetEncoder()
        encoder.fit([1, 2, 3])  # no-op
        tensor = encoder.to_tensor(encoder.load("ignored"))
        assert tensor.shape == ()
        assert encoder.num_classes is None

    def test_is_not_file_based_spatial_or_summarizable(self) -> None:
        from src.data.encoders import NullTargetEncoder
        from src.data.statistics import SupportsSummary

        encoder = NullTargetEncoder()
        assert encoder.file_based is False
        assert encoder.spatial is False
        assert not isinstance(encoder, SupportsSummary)

    def test_registered_under_null_key(self) -> None:
        from src.data.encoders import NullTargetEncoder
        from src.data.registry import target_encoders

        assert isinstance(target_encoders.create("null"), NullTargetEncoder)

    def test_dataset_target_with_none_column_needs_no_data_column(self, csv_path: Path) -> None:
        # A structure-only task (triplet/contrastive) declares no target column; the
        # dataset still yields a target so the step can index batch.targets[name].
        from src.data.encoders import NullTargetEncoder

        frame = pd.read_csv(csv_path)  # image_path + label, but no column for 'rank'
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[TargetBinding("rank", None, NullTargetEncoder())],
            transform=_make_transform((16, 16)),
        )
        sample = dataset[0]
        assert "rank" in sample.targets
        assert sample.targets["rank"].shape == ()
        assert "rank" not in sample.meta["target_sources"]  # not file-based → no recorded source


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
        assert codec.file_based is True  # read from a file (resolved against root_path)
        assert codec.spatial is True  # and rides through the geometric transform
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


class TestMaskEncoderIndexValidation:
    """Masks must arrive as class indices; out-of-range pixel values fail loudly at load."""

    @staticmethod
    def _write_mask(path: Path, values: np.ndarray) -> Path:
        mask_path = path / "mask.png"
        cv2.imwrite(str(mask_path), values.astype(np.uint8))
        return mask_path

    def test_out_of_range_pixel_values_raise_actionable_error(self, tmp_path: Path) -> None:
        from src.data import MaskEncoder

        raw = np.array([[0, 255]], dtype=np.uint8)  # {0, 255} PNG fed to a 2-class index encoder
        codec = MaskEncoder(class_mapping={0: "background", 1: "defect"})
        with pytest.raises(ValueError, match="class indices"):
            codec.load(self._write_mask(tmp_path, raw))

    def test_plain_index_mask_passes_unchanged(self, tmp_path: Path) -> None:
        from src.data import MaskEncoder

        raw = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        codec = MaskEncoder(class_mapping={0: "background", 1: "defect"})
        mask = codec.load(self._write_mask(tmp_path, raw))
        assert np.array_equal(mask, raw)

    def test_without_class_mapping_values_are_not_validated(self, tmp_path: Path) -> None:
        """No class count known at the encoder -> validation is the task config's job."""
        from src.data import MaskEncoder

        raw = np.array([[0, 255]], dtype=np.uint8)
        mask = MaskEncoder().load(self._write_mask(tmp_path, raw))
        assert np.array_equal(mask, raw)
