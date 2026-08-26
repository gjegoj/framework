"""Categorical targets: one class per cell (``label``), or several (``multilabel``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.encoders import LabelTargetEncoder, MultiLabelTargetEncoder
from src.data.registry import target_encoder_registry


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


def test_duplicate_declared_names_are_refused_at_the_encoder_too() -> None:
    """A copy-paste typo must not silently shrink the model: {0: a, 1: a} is one class in disguise."""
    with pytest.raises(ValueError, match="duplicated"):
        LabelTargetEncoder(classes={0: "a", 1: "a"})


def test_a_multilabel_vocabulary_is_declared_the_same_way() -> None:
    encoder = MultiLabelTargetEncoder(classes={0: "indoor", 1: "people"})

    with pytest.raises(LookupError, match="outdor"):
        encoder.fit(["indoor,people", "outdor"])


def fitted(values: list[object], **kwargs: object) -> MultiLabelTargetEncoder:
    encoder = MultiLabelTargetEncoder(**kwargs)  # type: ignore[arg-type]
    encoder.fit(values)
    return encoder


def test_labels_become_an_indicator_vector() -> None:
    encoder = fitted(["cat,dog", "bird", "cat"])

    assert encoder.class_names == ["bird", "cat", "dog"]
    assert encoder.encode("cat,dog").tolist() == [0.0, 1.0, 1.0]


def test_the_vector_is_float_because_the_loss_compares_probabilities() -> None:
    """Binary cross-entropy takes float targets, so nothing downstream has to cast."""
    encoded = fitted(["cat,dog"]).encode("cat")

    assert encoded.dtype == np.float32


def test_a_row_with_no_labels_is_all_zeros() -> None:
    """A negative example carries information in multi-label; it is not missing data."""
    encoder = fitted(["cat,dog", "cat", None])

    assert encoder.encode(None).tolist() == [0.0, 0.0]
    assert encoder.encode("").tolist() == [0.0, 0.0]


def test_lists_are_accepted_as_they_arrive_from_json_tables() -> None:
    encoder = fitted([["cat", "dog"], ["bird"]])

    assert encoder.class_names == ["bird", "cat", "dog"]
    assert encoder.encode(["bird", "dog"]).tolist() == [1.0, 0.0, 1.0]


def test_surrounding_spaces_do_not_create_separate_classes() -> None:
    encoder = fitted(["cat, dog", "dog,cat"])

    assert encoder.class_names == ["cat", "dog"]


def test_a_custom_separator_is_honoured() -> None:
    encoder = fitted(["cat|dog", "bird"], separator="|")

    assert encoder.class_names == ["bird", "cat", "dog"]
    assert encoder.encode("cat|bird").tolist() == [1.0, 1.0, 0.0]


def test_declared_classes_pin_the_order_and_skip_fitting() -> None:
    """Column order is what a deployed model's outputs mean; sometimes it must not drift."""
    encoder = MultiLabelTargetEncoder(classes={0: "dog", 1: "cat"})

    assert encoder.class_names == ["dog", "cat"]
    assert encoder.num_classes == 2
    assert encoder.encode("cat").tolist() == [0.0, 1.0]


def test_declared_classes_survive_a_fit_over_a_subset() -> None:
    """A rare class absent from a small slice must not reshuffle the index space."""
    encoder = MultiLabelTargetEncoder(classes={0: "dog", 1: "cat"})
    encoder.fit(["cat"])

    assert encoder.class_names == ["dog", "cat"]


def test_a_label_missing_from_the_vocabulary_is_reported() -> None:
    """The model has no output for it, so encoding it quietly would train on a lie."""
    encoder = fitted(["cat,dog"])

    with pytest.raises(LookupError, match="unicorn"):
        encoder.encode("cat,unicorn")


def test_encoding_before_fitting_is_reported() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        MultiLabelTargetEncoder().encode("cat")


def test_the_encoder_is_reachable_from_config_by_name() -> None:

    assert isinstance(target_encoder_registry.create("multilabel"), MultiLabelTargetEncoder)
