"""The log-key grammar has one owner: composition helpers and tokens."""

from __future__ import annotations

from src.core import Stage, log_keys


def test_join_composes_segments_with_the_separator() -> None:
    assert log_keys.join("val", "label", "ce") == "val/label/ce"


def test_total_loss_is_the_stage_prefixed_loss_key() -> None:
    assert log_keys.total_loss(Stage.TRAIN) == "train/loss"


def test_stage_members_compose_as_their_string_values() -> None:
    assert log_keys.join(Stage.VAL, "label", "accuracy") == "val/label/accuracy"


def test_a_stage_first_key_splits_into_series_by_stage() -> None:
    """One graph per series, stages as its lines — losses and metrics alike, no special case."""
    assert log_keys.split_for_tracker("val/label/f1") == ("label/f1", "val")
    assert log_keys.split_for_tracker("train/loss") == ("loss", "train")


def test_the_classes_of_one_metric_share_a_graph_with_their_mean() -> None:
    """Comparing classes is what a per-class metric is for, and stages cannot be that comparison.

    Split by stage instead, a forty-class run draws forty graphs of one line each.
    """
    assert log_keys.split_for_tracker("val/label/f1/cat") == ("val/label/f1", "cat")
    assert log_keys.split_for_tracker("val/label/f1/dog") == ("val/label/f1", "dog")
    assert log_keys.split_for_tracker("val/label/f1/mean") == ("val/label/f1", "mean")
    assert log_keys.split_for_tracker("train/label/f1/cat") == ("train/label/f1", "cat")


def test_a_loss_part_is_not_mistaken_for_a_per_class_leaf() -> None:
    """A part is scoped exactly once, so it stays two segments deep and keeps its stages together.

    A criterion that scoped twice would land its parts on a graph per stage, which
    is why the depth is asserted rather than assumed.
    """
    assert log_keys.split_for_tracker("train/label/ce") == ("label/ce", "train")
    assert log_keys.split_for_tracker("val/label/kl") == ("label/kl", "val")


def test_a_key_of_one_segment_stands_alone() -> None:
    """Nothing to compare it with and no leaf to name a line by."""
    assert log_keys.split_for_tracker("epoch") == ("epoch", "value")


def test_a_stage_less_family_shares_a_graph_by_its_leaves() -> None:
    """The learning rates of every parameter group belong on one graph, one line each.

    Did the head move faster than the encoder is the comparison a per-group rate is
    declared for, and a title per group is exactly the comparison it cannot make.
    """
    assert log_keys.split_for_tracker("lr/backbone") == ("lr", "backbone")
    assert log_keys.split_for_tracker("lr/label") == ("lr", "label")
