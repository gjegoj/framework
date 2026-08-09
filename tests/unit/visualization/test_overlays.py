"""What the page draws as pixels: the inlined image, and the two-rim mask over it."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from src.visualization.overlays import mask_overlay_uri, png_data_uri


def decode(uri: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert uri.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(uri.removeprefix(prefix))))


def pixels_of(uri: str) -> np.ndarray:
    return np.array(decode(uri))


def test_three_channels_encode_as_rgb_and_four_as_rgba() -> None:
    """The renderer inlines opaque images through the same call the overlays use."""
    assert decode(png_data_uri(np.zeros((4, 4, 3), dtype=np.uint8))).mode == "RGB"
    assert decode(png_data_uri(np.zeros((4, 4, 4), dtype=np.uint8))).mode == "RGBA"


def test_pixels_survive_the_round_trip() -> None:
    """A self-contained page shows what it was given; a lossy hop would silently recolour it."""
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    assert np.array_equal(pixels_of(png_data_uri(pixels)), pixels)


def test_fill_rim_and_outer_rim_land_where_morphology_says() -> None:
    """The look IoU-by-eye depends on: translucent fill, near-opaque rims, black outside."""
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    rgba = pixels_of(mask_overlay_uri(mask, (255, 0, 0)))

    assert rgba[4, 4].tolist() == [255, 0, 0, 78]  # interior: translucent class colour
    assert rgba[2, 4].tolist() == [255, 0, 0, 235]  # inner rim: near-opaque class colour
    assert rgba[1, 4].tolist() == [0, 0, 0, 235]  # outer rim: near-opaque black
    assert rgba[0, 4, 3] == 0  # beyond the rim: fully transparent


def test_a_mask_touching_the_edge_draws_no_rim_on_the_far_side() -> None:
    """``np.roll`` wraps a shape's neighbours onto the opposite edge, painting a stripe there.

    The far *corner* would not catch it: rolling down carries row 0 to the last
    row at the same columns, so the wrapped rim lands under the shape's own
    columns, not diagonally across the image.
    """
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:3, 0:3] = True

    rgba = pixels_of(mask_overlay_uri(mask, (0, 255, 0)))

    assert rgba[5, 0:3, 3].tolist() == [0, 0, 0]  # the far row, under the shape's columns
    assert rgba[0:3, 5, 3].tolist() == [0, 0, 0]  # the far column, beside the shape's rows


def test_a_shape_running_off_the_frame_keeps_its_rim_along_the_frame() -> None:
    """``beyond_the_edge`` is a decision, not an accident: the shape stays outlined.

    A cell cropped through an object would otherwise show a mask that fades into
    the image border, and the class it belongs to would be unreadable there.
    """
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:3, 0:3] = True

    rgba = pixels_of(mask_overlay_uri(mask, (0, 255, 0)))

    assert rgba[0, 1].tolist() == [0, 255, 0, 235]  # inner rim along the top edge
    assert rgba[1, 1].tolist() == [0, 255, 0, 78]  # and the fill still behind it


def test_a_picture_is_bounded_before_it_is_inlined() -> None:
    """Measured: eight cells of a ten-class segmentation at 512px inline to a 49 MB page."""
    pixels = np.zeros((600, 300, 3), dtype=np.uint8)

    assert decode(png_data_uri(pixels, max_side=200)).size == (100, 200)  # aspect kept
    assert decode(png_data_uri(pixels)).size == (300, 600)  # uncapped by default


def test_a_downscaled_mask_stays_a_mask() -> None:
    """Interpolating a boolean overlay would leave half-transparent pixels along every edge."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    rgba = pixels_of(mask_overlay_uri(mask, (255, 0, 0), max_side=16))

    assert rgba.shape[:2] == (16, 16)
    assert set(np.unique(rgba[..., 3])) <= {0, 78, 235}  # only the alphas we paint


def test_a_downscaled_mask_keeps_a_rim_on_every_side() -> None:
    """A rim is drawn at display size because nearest sampling cannot thin one that exists.

    Rimming first and shrinking after keeps whichever source rows the sampler
    lands on, and a 1px rim is exactly one row wide — measured, a square shrunk
    2x kept its top edge and lost its bottom one, so the mask read as leaking off
    the object rather than as a resampling artifact.
    """
    mask = np.zeros((512, 512), dtype=bool)
    mask[106:406, 106:406] = True

    rgba = pixels_of(mask_overlay_uri(mask, (255, 0, 0), max_side=256))

    outer = (rgba[..., 3] == 235) & (rgba[..., 0] == 0)
    rows, columns = np.where(outer)
    # A 150x150 square shown at half size: four sides of 150 pixels each.
    assert outer.sum() == 600
    assert (rows == rows.min()).sum() == 150  # top
    assert (rows == rows.max()).sum() == 150  # bottom
    assert (columns == columns.min()).sum() == 150  # left
    assert (columns == columns.max()).sum() == 150  # right
