"""Renderers: every kind has one, and each one's two halves hold separately."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from src.visualization.entities import (
    Classification,
    Classifications,
    Image,
    Label,
    Media,
    Regression,
    Segmentation,
    SegmentationClass,
    Text,
)
from src.visualization.registry import label_renderer_registry, media_renderer_registry
from src.visualization.renderers import FieldContext, leaves_of, render_label, render_media


def context(**colors: str) -> FieldContext:
    return FieldContext(task="t", kind="pred", colors=colors)


def test_every_label_kind_has_a_registered_renderer() -> None:
    """The exhaustiveness pin: a new entity cannot reach a run unrendered."""
    for kind in get_args(Label.__value__):
        assert kind in label_renderer_registry, kind.__name__


def test_every_media_kind_has_a_registered_renderer() -> None:
    for kind in get_args(Media.__value__):
        assert kind in media_renderer_registry, kind.__name__


def test_an_unregistered_kind_is_refused_with_the_registered_list() -> None:
    """The message is the registry's own — nothing hand-written to fall stale."""
    with pytest.raises(LookupError, match="Registered:"):
        render_label(object(), context())  # type: ignore[arg-type]


def test_a_classification_is_one_chip_with_its_own_colour() -> None:
    items = render_label(Classification("cat", confidence=0.9), context(cat="#ff0000"))

    assert len(items) == 1
    assert items[0].leaf == "cat"
    assert items[0].zone == "chips"
    assert items[0].color == "#ff0000"
    assert "cat 0.90" in items[0].overlay


def test_a_segmentation_is_one_cover_layer_per_class() -> None:
    label = Segmentation(
        classes=(
            SegmentationClass("pet", np.ones((4, 4), dtype=bool)),
            SegmentationClass("sky", np.zeros((4, 4), dtype=bool)),
        )
    )

    items = render_label(label, context())

    assert [item.leaf for item in items] == ["pet", "sky"]
    assert all(item.zone == "cover" for item in items)
    assert leaves_of(label) == ("pet", "sky")


def test_leaves_and_render_agree_on_the_names() -> None:
    """The two halves of one renderer speak of the same classes — the property
    that used to depend on two dispatch tables staying in step by hand."""
    label = Classifications(classifications=(Classification("cat"), Classification("dog")))

    assert leaves_of(label) == ("cat", "dog")
    assert [item.leaf for item in render_label(label, context())] == ["cat", "dog"]


def test_a_regression_leaf_is_the_one_slot_a_palette_colours() -> None:
    assert leaves_of(Regression(value=3.5)) == ("value",)


def test_an_image_fills_the_frame_with_its_own_aspect() -> None:
    item = render_media(Image(pixels=np.zeros((10, 20, 3), dtype=np.uint8)), alias="image")

    assert item.zone == "frame"
    assert item.aspect == 2.0


def test_a_text_is_a_captioned_strip_without_geometry() -> None:
    item = render_media(Text(text="a red car"), alias="caption")

    assert item.zone == "caption"
    assert item.aspect is None
    assert "a red car" in item.markup


def test_gt_ink_is_chosen_by_measured_contrast_not_fixed_white() -> None:
    """Fixed white ink read at ~1.9:1 on the palette's light classes; the ink flips.

    Both cases asserted, so the chooser cannot degenerate to either constant."""
    light = render_label(Classification("sun"), FieldContext("t", "gt", {"sun": "#d0a439"}))[0].overlay
    dark = render_label(Classification("sky"), FieldContext("t", "gt", {"sky": "#394cd0"}))[0].overlay

    assert "color:#10131a" in light
    assert "color:#ffffff" in dark
