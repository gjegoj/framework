"""``TargetEncoder`` contracts: fit/encode, raw values, facts exposed after fit."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data import LabelTargetEncoder, MaskTargetEncoder, MultiLabelTargetEncoder, ScalarTargetEncoder
from src.data.registry import target_encoder_registry


def write_mask(path: Path, classes: np.ndarray) -> Path:
    """Write a class-index mask as a grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), classes.astype(np.uint8))
    return path


def test_label_encoder_learns_sorted_vocabulary_on_fit() -> None:
    encoder = LabelTargetEncoder()

    encoder.fit(pd.Series(["dog", "cat", "dog"]))

    assert encoder.num_classes == 2
    assert encoder.class_names == ["cat", "dog"]


def test_label_encoder_encodes_to_a_raw_class_index() -> None:
    """Encoders stay raw: tensors are made once, by the transform or collation."""
    encoder = LabelTargetEncoder()
    encoder.fit(pd.Series(["dog", "cat"]))

    encoded = encoder.encode("dog")

    assert encoded == 1
    assert isinstance(encoded, int)


def test_label_encoder_names_known_classes_for_unseen_value() -> None:
    encoder = LabelTargetEncoder()
    encoder.fit(pd.Series(["cat", "dog"]))

    with pytest.raises(LookupError, match="bird"):
        encoder.encode("bird")


def test_label_encoder_refuses_to_encode_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        LabelTargetEncoder().encode("cat")


def test_scalar_encoder_needs_no_fit_and_reports_no_classes() -> None:
    encoder = ScalarTargetEncoder()

    encoded = encoder.encode(3.5)

    assert encoded == pytest.approx(3.5)
    assert isinstance(encoded, float)
    assert encoder.num_classes is None
    assert encoder.class_names is None


def test_built_in_encoders_are_registered_for_config() -> None:
    assert set(target_encoder_registry) == {"label", "multilabel", "scalar", "mask", "gaussian_bins", "linear_bins"}
    assert isinstance(target_encoder_registry.create("mask", num_classes=2), MaskTargetEncoder)


def test_only_mask_targets_are_spatial() -> None:
    """Spatiality is what tells a transform which targets ride the image's geometry."""
    assert MaskTargetEncoder(num_classes=2).spatial is True
    assert LabelTargetEncoder().spatial is False
    assert ScalarTargetEncoder().spatial is False


def test_mask_encoder_reads_class_indices_as_an_integer_array(tmp_path: Path) -> None:
    classes = np.array([[0, 1], [2, 0]])
    path = write_mask(tmp_path / "mask.png", classes)

    encoded = MaskTargetEncoder(num_classes=3).encode(path)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (2, 2)
    assert encoded.dtype == np.int64
    assert encoded.tolist() == classes.tolist()


def test_mask_encoder_prepends_its_own_root(tmp_path: Path) -> None:
    write_mask(tmp_path / "masks" / "one.png", np.zeros((2, 2)))

    encoded = MaskTargetEncoder(num_classes=2, root=tmp_path / "masks").encode("one.png")

    assert encoded.shape == (2, 2)


def test_mask_encoder_reports_the_class_count_it_was_given() -> None:
    assert MaskTargetEncoder(num_classes=4).num_classes == 4


def test_mask_encoder_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.png"):
        MaskTargetEncoder(num_classes=2, root=tmp_path).encode("absent.png")


def test_a_declared_vocabulary_validates_the_data_instead_of_learning_it() -> None:
    """A typo row must fail loudly, not silently grow an 11th class."""
    encoder = LabelTargetEncoder(classes={0: "cat", 1: "dog"})

    with pytest.raises(LookupError, match="catt"):
        encoder.fit(["cat", "catt", "dog"])


def test_a_declared_class_absent_from_train_is_legal() -> None:
    """A rare class missing from a small slice must not reshuffle the index space."""
    encoder = LabelTargetEncoder(classes={0: "cat", 1: "dog", 2: "cow"})

    encoder.fit(["cat", "dog"])

    assert encoder.num_classes == 3
    assert encoder.encode("cow") == 2


def test_a_multilabel_vocabulary_is_declared_the_same_way() -> None:
    encoder = MultiLabelTargetEncoder(classes={0: "indoor", 1: "people"})

    with pytest.raises(LookupError, match="outdor"):
        encoder.fit(["indoor,people", "outdor"])


def test_a_mask_derives_its_count_from_declared_classes() -> None:
    encoder = MaskTargetEncoder(classes={0: "background", 1: "defect"})

    assert encoder.num_classes == 2
    assert encoder.class_names == ["background", "defect"]


def test_a_mask_refuses_disagreeing_count_and_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        MaskTargetEncoder(num_classes=3, classes={0: "background", 1: "defect"})


def test_a_mask_needs_a_count_from_somewhere() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        MaskTargetEncoder()


def test_duplicate_declared_names_are_refused_at_the_encoder_too() -> None:
    """A copy-paste typo must not silently shrink the model: {0: a, 1: a} is one class in disguise."""
    with pytest.raises(ValueError, match="duplicated"):
        LabelTargetEncoder(classes={0: "a", 1: "a"})
