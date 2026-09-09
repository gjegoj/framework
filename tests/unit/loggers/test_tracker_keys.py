"""A log key as ClearML's ``(title, series)``: one graph per title, one line per series."""

from __future__ import annotations

from src.loggers.clearml import split_for_tracker


def test_a_stage_first_key_splits_into_series_by_stage() -> None:
    """One graph per series, stages as its lines — losses and metrics alike, no special case."""
    assert split_for_tracker("val/label/f1") == ("label/f1", "val")
    assert split_for_tracker("train/loss") == ("loss", "train")


def test_the_classes_of_one_metric_share_a_graph_with_their_mean() -> None:
    """Comparing classes is what a per-class metric is for, and stages cannot be that comparison.

    Split by stage instead, a forty-class run draws forty graphs of one line each.
    """
    assert split_for_tracker("val/label/f1/cat") == ("val/label/f1", "cat")
    assert split_for_tracker("val/label/f1/dog") == ("val/label/f1", "dog")
    assert split_for_tracker("val/label/f1/mean") == ("val/label/f1", "mean")
    assert split_for_tracker("train/label/f1/cat") == ("train/label/f1", "cat")


def test_a_loss_part_is_not_mistaken_for_a_per_class_leaf() -> None:
    """A part is scoped exactly once, so it stays two segments deep and keeps its stages together.

    A criterion that scoped twice would land its parts on a graph per stage, which
    is why the depth is asserted rather than assumed.
    """
    assert split_for_tracker("train/label/ce") == ("label/ce", "train")
    assert split_for_tracker("val/label/kl") == ("label/kl", "val")


def test_a_key_of_one_segment_stands_alone() -> None:
    """Nothing to compare it with and no leaf to name a line by."""
    assert split_for_tracker("epoch") == ("epoch", "value")


def test_a_stage_less_family_shares_a_graph_by_its_leaves() -> None:
    """The learning rates of every parameter group belong on one graph, one line each.

    Did the head move faster than the encoder is the comparison a per-group rate is
    declared for, and a title per group is exactly the comparison it cannot make.
    """
    assert split_for_tracker("lr/backbone") == ("lr", "backbone")
    assert split_for_tracker("lr/label") == ("lr", "label")
