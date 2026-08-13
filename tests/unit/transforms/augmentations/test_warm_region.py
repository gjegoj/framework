"""``MaskedPlanckianJitter``: patchy warmth inside a mask, and what it averages to."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from albumentations.augmentations.pixel.functional import planckian_jitter

from src.core import Sample
from src.transforms import AlbumentationsTransform
from src.transforms.augmentations import COOLEST, WARMEST, MaskedPlanckianJitter

SIDE = 64
INSIDE = (slice(0, SIDE // 2), slice(None))
"""The top half, which the mask claims; the bottom half is what must not move."""

OUTSIDE = (slice(SIDE // 2, None), slice(None))

PATCHY = (3400, 4200)
"""A range near the warm end, where the red-over-blue ratio moves fast enough to see."""


def image() -> np.ndarray:
    """Mid grey, so a warmer red and a colder blue both have room to move."""
    return np.full((SIDE, SIDE, 3), 128, dtype=np.uint8)


def textured() -> np.ndarray:
    """Every channel value represented, so an equality check is about more than one number."""
    return np.random.default_rng(0).integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8)


def mask(filled: bool = True, whole: bool = False) -> np.ndarray:
    marked = np.zeros((SIDE, SIDE), dtype=np.uint8)
    if whole:
        marked[:] = 1
    elif filled:
        marked[INSIDE] = 1
    return marked


def warmed(
    seed: int = 1,
    filled: bool = True,
    whole: bool = False,
    pixels: np.ndarray | None = None,
    **kwargs: Any,
) -> Sample:
    transform = AlbumentationsTransform(
        [MaskedPlanckianJitter(mask_key="lesion", p=1.0, **kwargs)],
        spatial_targets=["lesion"],
        label_targets=["warmth"],
        seed=seed,
    )
    given = image() if pixels is None else pixels
    return transform(Sample(inputs={"image": given}, targets={"lesion": mask(filled, whole), "warmth": 0.0}))


def drawn(
    spread: int | tuple[int, int], seed: int = 7, region: np.ndarray | None = None, **kwargs: Any
) -> dict[str, Any]:
    """The parameters the transform drew, so an invariant can be read rather than inferred."""
    transform = MaskedPlanckianJitter(mask_key="lesion", spread=spread, p=1.0, **kwargs)
    transform.set_random_seed(seed)
    marked = mask() if region is None else region
    return transform.get_params_dependent_on_data({"shape": (SIDE, SIDE, 3)}, {"lesion": marked})


# --- what warming a region means -------------------------------------------------


def test_only_the_masked_region_is_warmed() -> None:
    """The whole point: the rest of the frame must not become a shortcut."""
    pixels = warmed(spread=800, temperature_range=PATCHY).inputs["image"]

    assert not np.array_equal(pixels[INSIDE], image()[INSIDE])
    assert np.array_equal(pixels[OUTSIDE], image()[OUTSIDE])


def test_the_warmed_region_is_yellower_than_it_was() -> None:
    """Warmth is red over blue, not either alone — a check on the direction of the shift."""
    pixels = warmed(temperature_range=(WARMEST, WARMEST)).inputs["image"].astype(float)

    inside = pixels[INSIDE]
    assert inside[..., 0].mean() > 128  # red lifted
    assert inside[..., 2].mean() < 128  # blue cut


def test_the_mask_leaves_as_it_arrived() -> None:
    """An ImageOnlyTransform reads a spatial target; it must not rewrite one."""
    assert np.array_equal(warmed(spread=800, temperature_range=PATCHY).targets["lesion"], mask())


# --- the label --------------------------------------------------------------------


def test_the_drawn_temperature_reaches_the_bound_label() -> None:
    assert warmed(temperature_range=(4200, 4200)).targets["warmth"] == pytest.approx(4200.0)


def test_the_draw_varies_and_stays_where_it_was_told_to() -> None:
    drawn_values = {warmed(seed=seed).targets["warmth"] for seed in range(20)}

    assert len(drawn_values) > 1, "a single value would mean the range is not being drawn from"
    assert all(WARMEST <= one <= COOLEST for one in drawn_values)


def test_the_label_is_the_regions_mean_temperature() -> None:
    """Read off the finished field rather than reported as drawn, so it cannot drift
    from what the region actually is."""
    for seed in range(6):
        params = drawn(spread=800, seed=seed, temperature_range=PATCHY)

        assert params["temperature"] == pytest.approx(float(params["field"].mean()))


def test_the_label_tells_the_truth_when_the_swing_is_clipped() -> None:
    """A centre against the table's floor loses the cold half of its swing. Reporting the
    drawn centre would then describe a region that does not exist — measured, a centre of
    3000 K with an 2000 K swing leaves a region whose mean is 3250 K.
    """
    params = drawn(spread=2000, seed=1, region=mask(whole=True), temperature_range=(WARMEST, WARMEST))

    assert params["field"].min() == pytest.approx(float(WARMEST))  # clipped at the table
    assert params["temperature"] > WARMEST  # and the label says so


def test_an_empty_mask_warms_nothing_and_says_so() -> None:
    """Labelling an unchanged image as warmed is the defect this guards, the same one
    ``RandomBorderCrop.min_crop`` guards from the other side."""
    sample = warmed(filled=False, spread=800)

    assert np.array_equal(sample.inputs["image"], image())
    assert sample.targets["warmth"] == pytest.approx(float(COOLEST))


# --- the swing ---------------------------------------------------------------------


def test_the_neutral_end_is_the_original_byte_for_byte() -> None:
    """6500 K has to *mean* not warmed — see ``COOLEST`` for what the vendor's table
    does instead. Even the widest tint must vanish there, or a model separates
    "labelled cool" from "skipped by p" on a 4% brightness pedestal."""
    given = textured()

    sample = warmed(whole=True, pixels=given.copy(), spread=0, tint=0.5, temperature_range=(COOLEST, COOLEST))

    assert np.array_equal(sample.inputs["image"], given)


def test_an_even_warm_follows_the_vendor_curve_from_the_neutral_anchor() -> None:
    """The departure from ``planckian_jitter`` is the anchor, not the curve: warming
    at T is the vendor's shift at T relative to its own at 6500. Measured per channel
    on a float image, where the arithmetic carries no rounding to hide behind."""
    given = np.full((SIDE, SIDE, 3), 0.5, dtype=np.float32)

    sample = warmed(whole=True, pixels=given.copy(), spread=0, temperature_range=(4200, 4200))

    ours = np.asarray(sample.inputs["image"])[0, 0] / 0.5
    vendor = planckian_jitter(given.copy(), 4200, mode="blackbody")[0, 0] / 0.5
    anchor = planckian_jitter(given.copy(), COOLEST, mode="blackbody")[0, 0] / 0.5
    assert np.allclose(ours, vendor / anchor, atol=1e-3)


def test_without_a_spread_the_whole_region_is_one_temperature() -> None:
    """A flat image warmed evenly stays flat, which is what 'even' has to mean."""
    inside = warmed(spread=0).inputs["image"][INSIDE]

    assert len(np.unique(inside.reshape(-1, 3), axis=0)) == 1


def test_with_a_spread_the_region_is_uneven() -> None:
    inside = warmed(spread=800, temperature_range=PATCHY).inputs["image"][INSIDE]

    assert len(np.unique(inside.reshape(-1, 3), axis=0)) > 1


def test_the_swing_is_shared_out_by_area_not_by_value() -> None:
    """The fix that made the patches visible at all.

    A plasma field's values cluster around their own middle: mapped straight onto the
    swing, 73% of the region landed in the middle third of the range and the extremes
    went to a handful of pixels — a region that reads as one flat intensity. Ranking by
    area gives each third of the range about a third of the pixels. Measured at 35/32/32.
    """
    for roughness in (0.3, 0.5, 0.9):
        field = drawn(spread=1200, region=mask(whole=True), roughness=roughness, temperature_range=PATCHY)["field"]
        placed = (field - field.min()) / np.ptp(field)
        thirds = [float((placed < 1 / 3).mean()), float(((placed >= 1 / 3) & (placed < 2 / 3)).mean())]

        assert all(0.25 < share < 0.42 for share in thirds), f"roughness {roughness}: {thirds}"


def test_a_small_region_gets_the_whole_swing() -> None:
    """``spread`` means the same thing whatever the region's size.

    The generator spans ``[0, 1]`` over the whole *image*, so a region that catches only
    a sliver of it used to realise a fraction of the swing — measured, 33% at 1% coverage.
    Equalising inside the mask is what makes the knob size-independent.
    """
    for side in (SIDE, SIDE // 4, 6):
        region = np.zeros((SIDE, SIDE), dtype=np.uint8)
        region[:side, :side] = 1
        field = drawn(spread=1000, region=region, temperature_range=PATCHY)["field"]

        assert np.ptp(field) > 900, f"{side}x{side} region realised only {np.ptp(field):.0f} K of 1000"


def test_pixels_run_past_the_declared_range_but_never_past_the_table() -> None:
    """``temperature_range`` bounds the label, not the pixels.

    Bounding both with one number is what used to squeeze a wide swing into the middle
    of the range, where the ratio is flat and nothing shows. The coefficient table is
    the real limit, and going past it is what would break the colour.
    """
    field = drawn(spread=1200, region=mask(whole=True), temperature_range=(3400, 3400))["field"]

    assert field.min() < 3400  # past what was declared
    assert field.min() >= WARMEST  # never past what is tabulated


def neighbour_correlation(roughness: float | tuple[float, float], seed: int = 7) -> float:
    """How alike neighbouring pixels are — which is what 'feature size' means here."""
    field = drawn(spread=1000, seed=seed, region=mask(whole=True), roughness=roughness, temperature_range=PATCHY)[
        "field"
    ]
    laid_out = field.reshape(SIDE, SIDE)  # a whole mask reads back in row-major order
    return float(np.corrcoef(laid_out[:, :-1].ravel(), laid_out[:, 1:].ravel())[0, 1])


def test_roughness_sets_how_fine_the_patches_are() -> None:
    """Low roughness is one broad gradient, high roughness a mottle."""
    assert neighbour_correlation(0.1) > neighbour_correlation(0.9)


def test_a_roughness_range_draws_a_fresh_texture_per_application() -> None:
    """One sample fades across, the next is mottled — a fixed 0.5 gives neither:
    measured over these same seeds it stays inside 0.86..0.92."""
    correlations = [neighbour_correlation((0.05, 0.95), seed=seed) for seed in range(1, 9)]

    assert min(correlations) < 0.6  # some draw came out fine-grained
    assert max(correlations) > 0.95  # and some a broad gradient


def test_a_spread_range_draws_a_fresh_width_per_application() -> None:
    """The point of declaring a range: one sample nearly even, the next swung wide."""
    swings = [
        float(f["field"].max() - f["field"].min())
        for seed in range(1, 9)
        if (f := drawn(spread=(0, 1200), seed=seed, temperature_range=PATCHY, region=mask(whole=True)))
    ]

    assert min(swings) < 400
    assert max(swings) > 900


def test_a_single_number_spread_is_that_width_every_time() -> None:
    """The pre-range contract, kept: a plain int never became a lottery."""
    swings = [
        float(f["field"].max() - f["field"].min())
        for seed in range(1, 6)
        if (f := drawn(spread=1000, seed=seed, temperature_range=PATCHY, region=mask(whole=True)))
    ]

    assert all(900 <= swing <= 1000 for swing in swings)


def test_the_tint_varies_the_hue_at_one_temperature() -> None:
    """Two draws at the same kelvin come out different shades — the whole point of it."""
    first = warmed(seed=3, tint=0.3, temperature_range=(3500, 3500), spread=0).inputs["image"][INSIDE]
    second = warmed(seed=4, tint=0.3, temperature_range=(3500, 3500), spread=0).inputs["image"][INSIDE]

    assert not np.array_equal(first, second)


def test_the_tint_fades_as_the_region_cools() -> None:
    """A cast is the illumination's: the same gains that recolour a 3400 K region must
    barely graze a 6400 K one, or 'almost neutral' arrives visibly repainted."""

    def cast(temperature: int) -> int:
        plain = warmed(seed=5, tint=0.0, temperature_range=(temperature, temperature), spread=0)
        tinted = warmed(seed=5, tint=0.4, temperature_range=(temperature, temperature), spread=0)
        return int(np.abs(plain.inputs["image"].astype(int) - tinted.inputs["image"].astype(int)).max())

    assert cast(6400) <= 2
    assert cast(3400) > 20


def test_the_label_does_not_report_the_tint() -> None:
    """The tint is nuisance variation by design; kelvin stays about warmth alone."""
    plain = warmed(seed=6, tint=0.0, temperature_range=PATCHY, spread=600).targets["warmth"]
    tinted = warmed(seed=6, tint=0.5, temperature_range=PATCHY, spread=600).targets["warmth"]

    assert plain == tinted


def test_the_field_carries_one_temperature_per_warmed_pixel() -> None:
    """Sized by the region rather than the frame, so a small lesion costs a small pass."""
    region = mask()
    params = drawn(spread=800, region=region, temperature_range=PATCHY)

    assert len(params["field"]) == int(region.sum())


def test_an_image_too_small_to_vary_across_is_warmed_evenly() -> None:
    """One pixel has no inside for warmth to vary across, and the vendor's generator
    raises ``OverflowError`` when asked for a ``(1, 1)`` field."""
    transform = AlbumentationsTransform(
        [MaskedPlanckianJitter(mask_key="lesion", spread=800, p=1.0)],
        spatial_targets=["lesion"],
        label_targets=["warmth"],
    )
    sample = Sample(
        inputs={"image": np.full((1, 1, 3), 128, np.uint8)},
        targets={"lesion": np.ones((1, 1), np.uint8), "warmth": 0.0},
    )

    assert transform(sample).inputs["image"].shape == (1, 1, 3)


# --- refusals ----------------------------------------------------------------------


def test_a_mask_this_pipeline_was_not_given_is_named() -> None:
    transform = AlbumentationsTransform(
        [MaskedPlanckianJitter(mask_key="absent", p=1.0)],
        spatial_targets=["lesion"],
        label_targets=["warmth"],
    )

    with pytest.raises(KeyError, match="absent"):
        transform(Sample(inputs={"image": image()}, targets={"lesion": mask(), "warmth": 0.0}))


def test_a_mask_at_another_resolution_is_refused_by_shape() -> None:
    """Left to numpy this broadcasts or dies unhelpfully, deep inside a data loader."""
    transform = AlbumentationsTransform(
        [MaskedPlanckianJitter(mask_key="lesion", p=1.0)],
        spatial_targets=["lesion"],
        label_targets=["warmth"],
        is_check_shapes=False,
    )
    sample = Sample(inputs={"image": image()}, targets={"lesion": np.zeros((8, 8), np.uint8), "warmth": 0.0})

    with pytest.raises(ValueError, match="same resolution"):
        transform(sample)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"temperature_range": (6500, 3000)}, "low to high"),
        ({"temperature_range": (2000, 6500)}, "tabulated"),
        ({"temperature_range": (3000, 6500), "mode": "cied"}, "tabulated"),
        ({"spread": -1}, "cannot be negative"),
        ({"spread": (-5, 10)}, "cannot be negative"),
        ({"spread": (800, 200)}, "range of widths"),
        ({"roughness": 1.5}, "0 .* to 1"),
        ({"roughness": (0.9, 0.1)}, "range of them"),
        ({"roughness": (0.2, 1.4)}, "0 .* to 1"),
        ({"tint": -0.1}, "fractional gain swing"),
        ({"tint": 0.6}, "fractional gain swing"),
    ],
)
def test_a_setting_the_table_cannot_serve_is_refused_at_construction(kwargs: dict[str, Any], expected: str) -> None:
    """Failing here beats failing on the first batch, an hour into a run."""
    with pytest.raises(ValueError, match=expected):
        MaskedPlanckianJitter(mask_key="lesion", **kwargs)
