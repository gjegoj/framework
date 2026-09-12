"""The one thing this library does with pixels: put them in a page without a codec of its own."""

from __future__ import annotations

import numpy as np
import pytest

from src.visualization.png import data_uri, shrunk_to
from tests.support.pixels import decoded


class TestDataUri:
    def test_every_pixel_of_a_picture_survives_the_round_trip(self) -> None:
        pixels = np.random.default_rng(0).integers(0, 256, size=(7, 5, 3), dtype=np.uint8)

        assert np.array_equal(decoded(data_uri(pixels)), pixels)

    def test_an_overlay_keeps_the_transparency_it_was_drawn_with(self) -> None:
        pixels = np.random.default_rng(1).integers(0, 256, size=(4, 6, 4), dtype=np.uint8)

        assert np.array_equal(decoded(data_uri(pixels)), pixels)

    def test_a_picture_wider_than_the_bound_is_shrunk_to_it(self) -> None:
        pixels = np.full((40, 80, 3), 200, dtype=np.uint8)

        shown = decoded(data_uri(pixels, max_side=20))

        assert shown.shape == (10, 20, 3)

    def test_shrinking_averages_rather_than_samples(self) -> None:
        """Half black, half white, shrunk two-to-one across the divide: the seam reads mid-grey.

        Sampling one pixel of each pair would answer 0 or 255 and lose that a picture had
        detail there at all — which is exactly what a downscaled photograph must not do.
        """
        pixels = np.zeros((2, 4, 3), dtype=np.uint8)
        pixels[:, 1] = 255

        shown = decoded(data_uri(pixels, max_side=2))

        assert shown.shape == (1, 2, 3)
        assert shown[0, 0, 0] == pytest.approx(128, abs=1)

    def test_a_picture_inside_the_bound_is_left_alone(self) -> None:
        pixels = np.random.default_rng(2).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

        assert np.array_equal(decoded(data_uri(pixels, max_side=16)), pixels)


class TestShrunkTo:
    @pytest.mark.parametrize(
        ("width", "height", "max_side", "expected"),
        [
            (80, 40, 20, (20, 10)),
            (40, 80, 20, (10, 20)),
            (8, 8, 16, None),
            (8, 8, None, None),
            (3, 1, 2, (2, 1)),
        ],
    )
    def test_the_longest_side_decides_and_the_other_follows(
        self, width: int, height: int, max_side: int | None, expected: tuple[int, int] | None
    ) -> None:
        assert shrunk_to(width, height, max_side) == expected

    def test_a_side_never_rounds_away_to_nothing(self) -> None:
        """A very wide strip still has a row; zero would be a picture nobody can decode."""
        assert shrunk_to(1000, 1, 10) == (10, 1)


class TestRefusal:
    @pytest.mark.parametrize(
        "pixels",
        [
            pytest.param(np.zeros((4, 4, 3), dtype=np.float32), id="floats the pipeline never turned back into bytes"),
            pytest.param(np.zeros((4, 4), dtype=np.uint8), id="a plane with no channel axis"),
            pytest.param(np.zeros((4, 4, 2), dtype=np.uint8), id="two channels, which no PNG colour type is"),
        ],
    )
    def test_what_a_page_cannot_draw_is_named_rather_than_written_as_bytes(self, pixels: np.ndarray) -> None:
        """The one guard between a mis-shaped tensor and a page of undecodable bytes."""
        with pytest.raises(ValueError, match="uint8"):
            data_uri(pixels)
