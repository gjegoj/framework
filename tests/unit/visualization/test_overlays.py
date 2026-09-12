"""A mask over a picture: enough colour to read the class, enough edge to see its shape."""

from __future__ import annotations

import numpy as np

from src.visualization.overlays import FILL_ALPHA, RIM_ALPHA, mask_overlay_uri
from tests.support.pixels import decoded

RED = (200, 40, 40)

TRANSPARENT = 0


def block(size: int, top: int, left: int, side: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


class TestMaskOverlay:
    def test_inside_is_the_class_colour_seen_through(self) -> None:
        """Enough to read which class claimed the pixels, little enough to still see them."""
        drawn = decoded(mask_overlay_uri(block(9, 3, 3, 3), RED))

        assert tuple(drawn[4, 4]) == (*RED, FILL_ALPHA)

    def test_the_shape_is_edged_in_its_own_colour_inside_and_in_black_outside(self) -> None:
        """Two rims, because one is not enough: a class colour vanishes against a picture of that colour."""
        drawn = decoded(mask_overlay_uri(block(9, 3, 3, 3), RED))

        assert tuple(drawn[3, 3]) == (*RED, RIM_ALPHA)
        assert tuple(drawn[2, 3]) == (0, 0, 0, RIM_ALPHA)

    def test_a_pixel_no_rim_reaches_is_left_alone(self) -> None:
        drawn = decoded(mask_overlay_uri(block(9, 3, 3, 3), RED))

        assert drawn[0, 0, 3] == TRANSPARENT

    def test_a_shape_running_off_one_edge_is_not_edged_along_the_far_one(self) -> None:
        """Neighbours are found by padding rather than by rolling: the top row's are above it, not below."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[:2] = True

        drawn = decoded(mask_overlay_uri(mask, RED))

        assert np.all(drawn[5, :, 3] == TRANSPARENT)
        assert np.all(drawn[0, :, 3] == RIM_ALPHA)

    def test_the_shape_is_shrunk_before_it_is_edged_so_the_edge_survives(self) -> None:
        """Edging first and shrinking after drops whole rows of the rim, because sampling keeps one of two.

        An 8x8 block inside 32x32 becomes 4x4 at a bound of 16, whose every pixel but the middle four
        touches the outside — twelve rim pixels and four of fill. Edged at full size instead, only the
        rows and columns that survived the sampling would carry one.
        """
        drawn = decoded(mask_overlay_uri(block(32, 8, 8, 8), RED, max_side=16))

        assert drawn.shape == (16, 16, 4)
        assert int(np.sum((drawn[..., 3] == RIM_ALPHA) & (drawn[..., 0] == RED[0]))) == 12
        assert int(np.sum(drawn[..., 3] == FILL_ALPHA)) == 4


def test_the_fill_is_faint_beside_the_rim_it_is_drawn_with() -> None:
    """Both numbers are read by the tests above through the constants, so only their relation is left
    to hold: a fill as solid as the rim covers the picture it was drawn over."""
    assert 0 < FILL_ALPHA < RIM_ALPHA // 2


def test_a_shape_one_pixel_wide_survives_being_shrunk() -> None:
    """Taking one pixel per block loses it, and the class keeps its sidebar row while drawing nothing."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[:, 7] = True

    drawn = decoded(mask_overlay_uri(mask, RED, max_side=8))

    assert drawn[..., 3].any()
