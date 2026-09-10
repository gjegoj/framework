"""Encoders turn a raw cell into a tensor in two halves: load before augmentation, encode after."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.core import Axis, Geometry, Modality, Normalization, TargetInfo, require_shape, require_tensor
from src.data import Encoder, InputEncoder, TargetEncoder
from src.data.encoders import (
    GaussianBinsEncoder,
    ImageEncoder,
    LabelEncoder,
    LinearBinsEncoder,
    MaskEncoder,
    MultilabelEncoder,
    ScalarEncoder,
)
from src.data.encoders.continuous import BinnedEncoder
from src.data.registry import input_encoder_registry, target_encoder_registry
from tests.support.declarations import CLASSES


class TestContract:
    @pytest.mark.parametrize("name", list(target_encoder_registry))
    def test_every_registered_target_encoder_is_one(self, name: str) -> None:
        cls = target_encoder_registry.get(name)

        assert issubclass(cls, TargetEncoder)
        assert isinstance(cls.takes_classes, bool) and isinstance(cls.geometry, Geometry)

    @pytest.mark.parametrize("name", list(input_encoder_registry))
    def test_every_registered_input_encoder_is_one(self, name: str) -> None:
        assert issubclass(input_encoder_registry.get(name), InputEncoder)

    def test_load_fit_and_validate_are_optional_hooks(self) -> None:
        class Identity(Encoder):
            def encode(self, value: object) -> torch.Tensor:
                return torch.as_tensor(value)

        encoder = Identity()
        encoder.validate([1])

        assert encoder.load(3) == 3 and encoder.fit([1, 2]) is encoder


class TestLabel:
    @pytest.mark.parametrize(
        ("cell", "index"), [("dog", 1), (1, 1), ("cat", 0), (" cat ", 0)], ids=["name", "index", "first", "padded"]
    )
    def test_encodes_a_name_or_an_index_to_its_declared_position(
        self, label_encoder: LabelEncoder, cell: object, index: int
    ) -> None:
        encoded = require_tensor(label_encoder.encode(cell), name="label")

        assert encoded.item() == index and encoded.dtype is torch.long

    def test_reports_the_declared_vocabulary_as_its_info(self, label_encoder: LabelEncoder) -> None:
        assert label_encoder.info == TargetInfo(classes=CLASSES)

    @pytest.mark.parametrize("operation", ["encode", "validate", "fit"])
    def test_a_value_outside_the_vocabulary_is_refused_by_name_and_never_learned(
        self, label_encoder: LabelEncoder, operation: str
    ) -> None:
        argument = "bird" if operation == "encode" else ["cat", "bird"]

        with pytest.raises(LookupError, match="bird"):
            getattr(label_encoder, operation)(argument)


class TestMultilabel:
    @pytest.mark.parametrize("cell", ["cat,dog", ["cat", "dog"], " cat , dog "], ids=["string", "list", "padded"])
    def test_encodes_every_spelling_of_several_labels_to_one_indicator(self, cell: object) -> None:
        assert MultilabelEncoder(classes=CLASSES).encode(cell).tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("cell", ["", None, float("nan"), []], ids=["empty", "none", "nan", "empty list"])
    def test_an_empty_cell_is_a_negative(self, cell: object) -> None:
        assert MultilabelEncoder(classes=CLASSES).encode(cell).tolist() == [0.0, 0.0]

    def test_a_custom_separator_is_honoured(self) -> None:
        assert (
            require_tensor(MultilabelEncoder(classes=CLASSES, separator="|").encode("cat|dog"), name="tags").sum() == 2
        )


class TestScalarAndBins:
    def test_scalar_is_a_float_tensor(self) -> None:
        assert ScalarEncoder().encode("2.5").dtype is torch.float32 and ScalarEncoder().info == TargetInfo()

    @pytest.mark.parametrize("cls", [LinearBinsEncoder, GaussianBinsEncoder])
    def test_bins_learn_their_range_on_fit_and_report_bin_centres_as_values(self, cls: type[BinnedEncoder]) -> None:
        encoder = cls(bins=10).fit([0.0, 10.0])
        info = encoder.info

        assert info.num_classes == 10 and info.values is not None and len(info.values) == 10
        assert info.classes is not None and list(info.classes) == list(range(10))

    @pytest.mark.parametrize("cls", [LinearBinsEncoder, GaussianBinsEncoder])
    def test_a_declared_range_needs_no_fitting(self, cls: type[BinnedEncoder]) -> None:
        encoder = cls(bins=10, low=0.0, high=10.0)

        assert encoder.info.num_classes == 10
        assert torch.isclose(require_tensor(encoder.encode(5.0), name="bins").sum(), torch.tensor(1.0))

    def test_linear_bins_split_a_value_between_its_two_neighbours(self) -> None:
        encoder = LinearBinsEncoder(bins=3, low=0.0, high=2.0)  # centres at 0, 1, 2

        assert encoder.encode(0.25).tolist() == pytest.approx([0.75, 0.25, 0.0])

    def test_gaussian_bins_put_the_mode_on_the_nearest_centre(self) -> None:
        encoder = GaussianBinsEncoder(bins=10, low=0.0, high=10.0, sigma=0.5)

        assert int(encoder.encode(3.0).argmax()) == int(torch.tensor(encoder.info.values).sub(3.0).abs().argmin())

    @pytest.mark.parametrize(
        ("cls", "kwargs"),
        [
            (LinearBinsEncoder, {"bins": 1}),
            (LinearBinsEncoder, {"bins": 5, "low": 1.0}),
            (LinearBinsEncoder, {"bins": 5, "low": 2.0, "high": 1.0}),
            (GaussianBinsEncoder, {"bins": 5, "sigma": -1.0}),
            (GaussianBinsEncoder, {"bins": 4}),
        ],
        ids=["one bin", "half a range", "inverted range", "negative sigma", "too few bins for a learned sigma"],
    )
    def test_refuses_a_layout_it_cannot_serve(self, cls: type[BinnedEncoder], kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            cls(**kwargs)

    def test_refuses_to_learn_a_range_from_a_constant_or_empty_split(self) -> None:
        with pytest.raises(ValueError):
            LinearBinsEncoder(bins=5).fit([3.0, 3.0])
        with pytest.raises(ValueError):
            LinearBinsEncoder(bins=5).fit([])


GRAY = {"grayscale": True, "mean": (0.5,), "std": (0.5,)}


class TestImage:
    @pytest.mark.parametrize(("kwargs", "channels"), [({}, 3), (GRAY, 1)], ids=["rgb", "grayscale"])
    def test_declares_shape_modality_geometry_and_normalization(self, kwargs: dict[str, Any], channels: int) -> None:
        info = ImageEncoder(image_size=(224, 224), **kwargs).info

        shape = require_shape(info.shape, name="image")
        assert info.modality == Modality.IMAGE and ImageEncoder.geometry is Geometry.IMAGE
        assert shape.axes == (Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH) and shape.sizes == (channels, 224, 224)
        assert isinstance(info.normalization, Normalization) and len(info.normalization.mean) == channels

    @pytest.mark.parametrize(("kwargs", "shape"), [({}, (6, 8, 3)), (GRAY, (6, 8))], ids=["rgb", "grayscale"])
    def test_loads_rgb_pixels_or_one_gray_plane(
        self, images: Path, kwargs: dict[str, Any], shape: tuple[int, ...]
    ) -> None:
        loaded = ImageEncoder(image_size=(6, 8), root=images, **kwargs).load("a.png")

        assert isinstance(loaded, np.ndarray) and loaded.shape == shape and loaded.dtype == np.uint8

    def test_encode_accepts_only_the_tensor_the_stage_pipeline_made(self, images: Path) -> None:
        encoder = ImageEncoder(image_size=(6, 8), root=images)
        tensor = torch.zeros(3, 6, 8)

        assert encoder.encode(tensor) is tensor
        with pytest.raises(ValueError, match="transforms"):
            encoder.encode(encoder.load("a.png"))

    @pytest.mark.parametrize(
        "kwargs",
        [{"image_size": (0, 4)}, {"image_size": (4, 4, 4)}, {"image_size": (4, 4), "grayscale": True}],
        ids=["zero side", "three sides", "three means for one gray plane"],
    )
    def test_refuses_a_declaration_it_cannot_serve(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ImageEncoder(**kwargs)

    def test_identifies_a_file_by_its_resolved_path_so_roots_never_collide(self, images: Path, tmp_path: Path) -> None:
        assert ImageEncoder(image_size=(4, 4), root=images).cache_key("a.png") != ImageEncoder(
            image_size=(4, 4), root=tmp_path
        ).cache_key("a.png")
        assert LabelEncoder(classes=CLASSES).cache_key("cat") is None

    def test_a_missing_or_undecodable_file_is_named(self, images: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"nope\.png"):
            ImageEncoder(image_size=(4, 4), root=images).load("nope.png")


class TestMask:
    @pytest.mark.parametrize("as_tensor", [False, True], ids=["array", "int32 tensor as the pipeline returns it"])
    def test_loads_class_indices_and_encodes_a_long_tensor(self, mask_encoder: MaskEncoder, as_tensor: bool) -> None:
        loaded = mask_encoder.load("a_mask.png")

        mask = require_tensor(
            mask_encoder.encode(torch.as_tensor(loaded, dtype=torch.int32) if as_tensor else loaded), name="mask"
        )

        assert loaded.dtype == np.int64 and loaded.shape == (6, 8)
        assert mask.dtype is torch.long and mask.shape == (6, 8) and int(mask.sum()) == 4 * 5
        assert MaskEncoder.geometry is Geometry.MASK and MaskEncoder.takes_classes

    def test_validate_refuses_a_pixel_outside_the_vocabulary(self, images: Path) -> None:
        encoder = MaskEncoder(classes={0: "background"}, root=images)

        with pytest.raises(ValueError, match="1"):
            encoder.validate(["a_mask.png"])
