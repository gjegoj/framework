"""The visualization IR: framework-agnostic media and labels a renderer can draw."""

from __future__ import annotations

import numpy as np

from src.visualization import (
    Classification,
    Image,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Text,
    Verdict,
)


def test_fields_are_keyed_structurally_not_by_glued_strings() -> None:
    """A task named with an underscore broke the reference's rpartition round-trip."""
    view = SampleView()

    view.fields[("my_task", "gt")] = Classification(label="cat")

    assert ("my_task", "gt") in view.fields


def test_a_sample_shows_every_input_it_has_not_only_the_first() -> None:
    """A CLIP-style run pairs an image with a caption; drawing one halves what it is about."""
    view = SampleView(
        media={
            "image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8), source="a.png"),
            "caption": Text(text="a cat on a mat"),
        }
    )

    assert set(view.media) == {"image", "caption"}


def test_a_verdict_without_a_binary_judgement_says_so() -> None:
    """A regression is never 'correct'; None is what keeps the mistakes filter honest."""
    assert Verdict(scores=(Score(name="mae", value=0.12),)).correct is None
    assert Verdict(correct=False).correct is False


def test_a_measured_number_stays_a_number() -> None:
    """A free-text summary could be printed but never filtered; a slider needs the value."""
    verdict = Verdict(scores=(Score(name="iou", value=0.62),))

    assert verdict.scores[0].value == 0.62


def test_a_task_can_measure_itself_more_than_one_way() -> None:
    """A segmentation sample has an IoU and a Dice; a singular field had nowhere to put the second."""
    verdict = Verdict(scores=(Score("iou", 0.62), Score("dice", 0.75)))

    assert [score.name for score in verdict.scores] == ["iou", "dice"]


def test_segmentation_carries_masks_without_equality() -> None:
    """An ndarray field breaks generated __eq__; nothing compares labels, so eq is off."""
    mask = np.ones((2, 2), dtype=bool)
    label = Segmentation(classes=(SegmentationClass("cat", mask),))

    assert label.classes[0].mask.shape == (2, 2)
