"""The log-key grammar has one owner: composition helpers and tokens."""

from __future__ import annotations

from src.core import Stage, log_keys


def test_join_composes_segments_with_the_separator() -> None:
    assert log_keys.join("val", "label", "ce") == "val/label/ce"


def test_total_loss_is_the_stage_prefixed_loss_key() -> None:
    assert log_keys.total_loss(Stage.TRAIN) == "train/loss"


def test_stage_members_compose_as_their_string_values() -> None:
    assert log_keys.join(Stage.VAL, "label", "accuracy") == "val/label/accuracy"


def test_a_stage_first_key_parses_into_its_stage_and_path() -> None:
    """One parser for the grammar, so no consumer re-splits a string by hand."""
    key = log_keys.parse("val/label/f1")

    assert key.stage == Stage.VAL
    assert key.path == ("label", "f1")
    assert not key.per_class


def test_a_per_class_leaf_is_recognised_by_its_depth_and_collapses_to_its_family() -> None:
    """``{task}/{metric}/{class}`` is the one shape a vector metric writes; nothing needs to know the value's meaning."""
    key = log_keys.parse("val/label/f1/cat")

    assert key.per_class
    assert key.leaf == "cat"
    assert key.family == "val/label/f1"
    assert not key.is_mean
    assert log_keys.parse("val/label/f1/mean").is_mean


def test_a_stage_less_key_has_no_stage() -> None:
    assert log_keys.parse("lr/backbone").stage is None
    assert log_keys.parse("epoch") == log_keys.LogKey(stage=None, path=("epoch",))
