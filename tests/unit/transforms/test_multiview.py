"""Views of one picture: the same sample drawn several times, stacked for an objective that compares them."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform, MultiViewTransform, Rotate90

VIEWS, SIDE = 2, 4


def geometries(**targets: Geometry) -> dict[str, dict[str, Geometry]]:
    return {
        "inputs": {"image": Geometry.IMAGE},
        "targets": {"label": Geometry.NONE, **targets},
        "auxiliary_inputs": {},
    }


def sample() -> Sample:
    """A picture whose halves differ, so a draw that moves it is observable, and a label that never moves."""
    image = np.zeros((SIDE, SIDE, 3), np.uint8)
    image[:, SIDE // 2 :] = 255
    return Sample(inputs={"image": image}, targets={"label": "cat"}, metadata={"row": 3})


def chain(*drawn: A.BasicTransform) -> AlbumentationsTransform:
    """A stage's own chain: whatever it draws, then the crossing into tensors that ends every one."""
    return AlbumentationsTransform([*drawn, ToTensorV2()], seed=0)


def viewed(views: int = VIEWS, *, base: AlbumentationsTransform | None = None, **targets: Geometry) -> Any:
    return MultiViewTransform(views, base=base or chain()).with_geometry(**geometries(**targets))


def test_the_viewed_input_arrives_as_a_stack_and_everything_else_is_left_where_it_was() -> None:
    """One picture becomes several; a label is the same answer for all of them and is not multiplied."""
    drawn = viewed()(sample())

    image = drawn.inputs["image"]
    assert isinstance(image, Tensor) and tuple(image.shape) == (VIEWS, 3, SIDE, SIDE)
    assert drawn.targets["label"] == "cat"
    assert drawn.metadata == {"row": 3}


def test_every_view_is_drawn_on_its_own() -> None:
    """Identical copies teach a contrastive objective nothing: what it learns from is the difference."""
    drawn = viewed(6, base=chain(A.HorizontalFlip(p=0.5)))(sample())

    views = drawn.inputs["image"]
    assert isinstance(views, Tensor)
    assert not all(torch.equal(views[0], one) for one in views[1:])


def test_a_second_value_that_moves_with_the_pixels_is_refused_by_name() -> None:
    """A mask drawn beside a view is thrown away with it, and the run trains on targets from another picture.

    Which is what the legacy this replaces did: it drew the whole sample and kept only the image, so a
    segmentation target stayed as it was loaded while every view had moved away from it.
    """
    with pytest.raises(ValueError, match="mask"):
        viewed(mask=Geometry.MASK)


def test_a_chain_whose_draw_is_the_answer_cannot_be_drawn_more_than_once() -> None:
    """`Rotate90` makes the turn the answer, and a sample has one place to keep it while views have many."""
    answering = AlbumentationsTransform([Rotate90(task="turn"), ToTensorV2()], seed=0)

    with pytest.raises(ValueError, match="turn"):
        MultiViewTransform(VIEWS, base=answering).with_geometry(**geometries(turn=Geometry.NONE))


def test_a_single_view_is_not_a_view() -> None:
    """The objective compares one draw against another; with one there is nothing on the other side."""
    with pytest.raises(ValueError, match="two"):
        MultiViewTransform(1, base=chain())
