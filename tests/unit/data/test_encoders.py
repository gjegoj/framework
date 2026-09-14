"""Encoders turn a raw cell into a tensor in two halves: load before augmentation, encode after."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.core import Axis, Geometry, Modality, Normalization, TargetInfo, TensorShape, require_tensor
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
from src.data.encoders.identity import IdentityEncoder
from src.data.encoders.text import TextEncoder
from src.data.registry import input_encoder_registry, target_encoder_registry
from tests.support.declarations import CLASSES
from tests.support.text import WORDS, text_family


class TestContract:
    @pytest.mark.parametrize("name", list(target_encoder_registry))
    def test_every_registered_target_encoder_is_one(self, name: str) -> None:
        cls = target_encoder_registry.get(name)

        assert issubclass(cls, TargetEncoder)
        assert isinstance(cls.takes_classes, bool) and isinstance(cls.geometry, Geometry)

    @pytest.mark.parametrize("name", list(input_encoder_registry))
    def test_every_registered_input_encoder_is_one(self, name: str) -> None:
        assert issubclass(input_encoder_registry.get(name), InputEncoder)

    def test_an_encoder_has_to_do_nothing_but_encode(self) -> None:
        """``load`` is where a cell becomes something a transform can move, and a value that already is
        one needs no preparing — so the default is to hand it back."""

        class Identity(Encoder):
            def encode(self, value: object) -> torch.Tensor:
                return torch.as_tensor(value)

        assert Identity().load(3) == 3

    def test_reading_a_split_is_asked_of_targets_and_of_nothing_else(self) -> None:
        """Only a target is ever fitted or validated: the preprocessor fits the training split's targets
        and checks the others against them, while an input is only ever encoded."""

        class Counted(TargetEncoder):
            @property
            def info(self) -> TargetInfo:
                return TargetInfo()

            def encode(self, value: object) -> torch.Tensor:
                return torch.as_tensor(value)

        encoder = Counted()
        encoder.validate([1])

        assert encoder.fit([1, 2]) is encoder
        assert not hasattr(InputEncoder, "fit"), "an input encoder is offered a hook nothing would call"


class TestLabel:
    @pytest.mark.parametrize(
        ("cell", "index"), [("dog", 1), (1, 1), ("cat", 0), (" cat ", 0)], ids=["name", "index", "first", "padded"]
    )
    def test_encodes_a_name_or_an_index_to_its_declared_position(
        self, label_encoder: LabelEncoder, cell: object, index: int
    ) -> None:
        encoded = require_tensor(label_encoder.encode(cell), name="label")

        assert encoded.item() == index and encoded.dtype is torch.long

    @pytest.mark.parametrize(
        "classes",
        [pytest.param({0: "1", 1: "0"}, id="swapped"), pytest.param({0: "cat", 1: "0"}, id="one of them")],
    )
    def test_a_vocabulary_whose_word_spells_another_class_is_refused(self, classes: dict[int, str]) -> None:
        """A column writes names or indices, so 'the label is 1' must have one answer."""
        with pytest.raises(ValueError, match="spell another class"):
            LabelEncoder(classes=classes)

    def test_a_class_may_be_named_after_its_own_index(self) -> None:
        """What a rotation task declares: four quarter turns named by the turns themselves."""
        assert LabelEncoder(classes={0: "0", 1: "1", 2: "2", 3: "3"}).position("2") == 2

    def test_a_class_declared_with_padding_is_still_found_by_its_word(self) -> None:
        """A cell is read stripped, so a vocabulary written with padding has to be read the same way."""
        assert int(require_tensor(LabelEncoder(classes={0: "cat ", 1: "dog"}).encode("cat"), name="label")) == 0

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


class TestIdentity:
    """A vocabulary the training split settles, and evaluation splits that are not held to it."""

    def test_it_learns_one_index_per_identity_the_training_split_showed(self) -> None:
        encoder = IdentityEncoder().fit(["bob", "ann", "bob", "cid"])

        assert encoder.info == TargetInfo(classes={0: "ann", 1: "bob", 2: "cid"}, open_set=True)

    def test_it_indexes_identities_in_an_order_no_row_order_can_change(self) -> None:
        """An objective keeps one prototype per index, a checkpoint holds them, and a run restores them.

        An order that followed the rows would pair a restored prototype with a different identity after a
        reshuffle, a different split seed or a different `max_samples` — a model that existed at no point.
        """
        assert IdentityEncoder().fit(["cid", "ann", "bob"]).info == IdentityEncoder().fit(["bob", "cid", "ann"]).info

    def test_an_identity_the_training_split_never_showed_is_encoded_rather_than_refused(self) -> None:
        """Holding identities out of training is the whole point; refusing them is what forbade it."""
        encoder = IdentityEncoder().fit(["ann", "bob"])

        encoder.validate(["dee", "eve"])

        given = [int(require_tensor(encoder.encode(name), name="t")) for name in ("dee", "eve")]

        assert given[0] != given[1], "two identities training never showed are still two"

    def test_an_identity_training_never_showed_never_takes_a_learned_ones_index(self) -> None:
        """They share one column, and a reading that confused the two would count a match that never was."""
        encoder = IdentityEncoder().fit(["ann", "bob"])
        encoder.validate(["dee", "eve"])

        assert {int(require_tensor(encoder.encode(name), name="t")) for name in ("dee", "eve")}.isdisjoint({0, 1})

    def test_what_it_publishes_is_what_training_settled_however_much_it_can_encode(self) -> None:
        """Reading a split widens what the encoder can encode; it never widens what the objective is sized by."""
        encoder = IdentityEncoder().fit(["ann", "bob"])

        encoder.validate(["dee", "eve", "fay"])

        assert encoder.info.num_classes == 2

    def test_an_identity_no_split_was_read_for_is_refused_rather_than_given_an_index_here(self) -> None:
        """Encoding happens once per sample, in whichever worker holds a copy of this encoder.

        Two workers inventing an index for the same unread identity would invent two, and the reading
        would count them as different pictures of different things.
        """
        encoder = IdentityEncoder().fit(["ann", "bob"])

        with pytest.raises(LookupError, match="dee"):
            encoder.encode("dee")

    def test_an_unfitted_encoder_names_what_would_need_no_fitting(self) -> None:
        with pytest.raises(ValueError, match="classes"):
            _ = IdentityEncoder().info

    def test_a_training_split_holding_no_identity_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            IdentityEncoder().fit([])

    @pytest.mark.parametrize("read", ["fit", "validate"], ids=["learning them", "checking a split"])
    def test_a_cell_naming_no_identity_is_refused_where_the_split_is_read(self, read: str) -> None:
        """A blank identity is not one identity and not none of them; whose it is, is a question for the data."""
        encoder = IdentityEncoder()
        if read == "validate":
            encoder.fit(["ann", "bob"])

        with pytest.raises(ValueError, match="names no identity"):
            getattr(encoder, read)(["ann", " "])


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

    @pytest.mark.parametrize("cls", [LinearBinsEncoder, GaussianBinsEncoder])
    def test_a_value_the_layout_cannot_hold_is_refused_rather_than_quietly_pulled_to_the_edge(
        self, cls: type[BinnedEncoder]
    ) -> None:
        """Measured: with bins over [0, 10], the target 100 encodes to the edge and reads back as 10,
        so a model predicting 10 scored a perfect error on a sample it was 90 away from."""
        with pytest.raises(ValueError, match="outside"):
            cls(bins=10, low=0.0, high=10.0).validate([5.0, 100.0])

    @pytest.mark.parametrize("cls", [LinearBinsEncoder, GaussianBinsEncoder])
    def test_a_declared_layout_is_checked_against_the_split_it_is_fitted_on_too(self, cls: type[BinnedEncoder]) -> None:
        """Learning the range from training data puts every training value inside it; declaring one does not."""
        with pytest.raises(ValueError, match="outside"):
            cls(bins=10, low=0.0, high=10.0).fit([5.0, 100.0])

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

        shape = info.shape
        assert isinstance(shape, TensorShape)
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

    def test_a_stack_of_views_is_the_declared_image_as_many_times(self, images: Path) -> None:
        """A stage may draw one picture several times, and what the input *is* stays what one view is.

        The view axis rides with the batch axis: no declaration carries it, so the check here reads the
        shape of a view off the end rather than demanding the whole of it.
        """
        encoder = ImageEncoder(image_size=(6, 8), root=images)
        views = torch.zeros(2, 3, 6, 8)

        assert encoder.encode(views) is views

    def test_a_tensor_with_more_in_front_of_it_than_views_is_still_refused(self, images: Path) -> None:
        """One leading axis is what a stage can make; anything deeper is a shape nobody in this run built."""
        with pytest.raises(ValueError, match="transforms"):
            ImageEncoder(image_size=(6, 8), root=images).encode(torch.zeros(2, 2, 3, 6, 8))

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


LENGTH = 8
"""How many tokens this fixture's captions become, short ones padded to it and long ones cut to it."""


class TestText:
    @pytest.fixture
    def encoder(self, tmp_path: Path) -> TextEncoder:
        return TextEncoder(str(text_family(tmp_path / "family")), max_length=LENGTH)

    def test_declares_one_length_for_every_tensor_the_family_reads(self, encoder: TextEncoder) -> None:
        """A text input is a tree — an id per token and whatever the family reads beside it — not one tensor."""
        declared = encoder.info.shape

        assert encoder.info.modality == Modality.TEXT
        assert isinstance(declared, Mapping)
        assert set(declared) == {"input_ids", "token_type_ids", "attention_mask"}
        assert all(one == TensorShape(axes=(Axis.TOKENS,), sizes=(LENGTH,)) for one in declared.values())

    def test_a_caption_encodes_to_exactly_the_tree_it_declared(self, encoder: TextEncoder) -> None:
        """What is published and what is produced are one statement; two would size a head for a tensor nobody makes."""
        encoded = encoder.encode("a brown dog")
        declared = encoder.info.shape

        assert isinstance(encoded, Mapping) and isinstance(declared, Mapping)
        assert set(encoded) == set(declared)
        assert all(require_tensor(value, name=name).shape == (LENGTH,) for name, value in encoded.items())
        assert all(require_tensor(value, name=name).dtype == torch.long for name, value in encoded.items())

    @pytest.mark.parametrize("caption", ["a", " ".join(WORDS[5:] * 3)], ids=["shorter", "longer"])
    def test_every_caption_arrives_at_the_declared_length(self, encoder: TextEncoder, caption: str) -> None:
        """The length is the declaration, so a batch stacks whatever the captions were."""
        encoded = encoder.encode(caption)

        assert isinstance(encoded, Mapping)
        assert require_tensor(encoded["input_ids"], name="input_ids").shape == (LENGTH,)

    def test_which_tokens_are_padding_travels_with_them(self, encoder: TextEncoder) -> None:
        """Padding is marked in the tree itself, so nothing downstream has to guess it from a pad id."""
        short = encoder.encode("a dog")
        full = encoder.encode(" ".join(WORDS[5:] * 3))

        assert isinstance(short, Mapping) and isinstance(full, Mapping)
        assert int(require_tensor(short["attention_mask"], name="mask").sum()) < LENGTH
        assert int(require_tensor(full["attention_mask"], name="mask").sum()) == LENGTH

    @pytest.mark.parametrize(
        "cell", [None, float("nan"), 42, "", "   "], ids=["none", "nan", "number", "empty", "blank"]
    )
    def test_a_cell_holding_no_caption_is_refused_by_name(self, encoder: TextEncoder, cell: object) -> None:
        """A column with a hole in it is named here, rather than read as the empty sentence by a tokenizer."""
        with pytest.raises(ValueError, match="no text to read"):
            encoder.encode(cell)

    @pytest.mark.parametrize("max_length", [0, -3])
    def test_a_length_that_holds_nothing_is_refused(self, tmp_path: Path, max_length: int) -> None:
        with pytest.raises(ValueError, match="max_length"):
            TextEncoder(str(text_family(tmp_path / "family")), max_length=max_length)

    def test_the_name_resolves_long_before_the_library_behind_it_is_needed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Importing the package registers every encoder; a run that reads no text pays for none of transformers."""
        monkeypatch.setitem(sys.modules, "transformers", None)

        assert issubclass(input_encoder_registry.get("text"), TextEncoder)
        with pytest.raises(ImportError):
            TextEncoder(str(tmp_path), max_length=LENGTH)
