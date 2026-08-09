"""Multi-label targets: several labels per row into one indicator vector."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import MultiLabelTargetEncoder


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
    from src.data.registry import target_encoder_registry

    assert isinstance(target_encoder_registry.create("multilabel"), MultiLabelTargetEncoder)
