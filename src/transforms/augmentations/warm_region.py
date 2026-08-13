"""Warmth confined to one region, patchy inside it, and the temperature it averages to."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import albumentations as A
import numpy as np
from albucore import MAX_VALUES_BY_DTYPE
from albumentations.augmentations.pixel.functional import PLANCKIAN_COEFFS, generate_plasma_pattern

if TYPE_CHECKING:
    from collections.abc import Mapping

type Mode = Literal["blackbody", "cied"]

WARMEST = 3000
"""The most yellow a blackbody shift goes, and so the low end of the default range.

Warmth here is the *ratio* between the red and blue multipliers, not either alone.
Measured on ``blackbody``: at 3000 K red is scaled 1.674 and blue 0.003, a ratio of
523 — blue is all but removed. At 6500 K the two are 1.046 and 1.039, a ratio of
1.007, so the hue barely moves and only the brightness lifts by about 4%.

That ratio is **exponential** in kelvin, which is the single most useful fact here:
400 K of swing is worth a ratio change of 0.35 around 4400 K and of 3.1 around 3600 K.
A region meant to look patchy has to be centred near the warm end.
"""

COOLEST = 6500
"""The high end of the default range, and the anchor every coefficient is normalised to.

The vendor's table is not neutral anywhere — at 6500 K it still multiplies red by
1.046 and blue by 1.039, a ~4% brightness pedestal on a region whose label says
"not warmed", and a shortcut a model can read. So the multipliers here are divided
by their own value at this anchor: at 6500 K they are exactly one, the region is
byte-for-byte the original, and a sample this transform leaves cool is
indistinguishable from one ``p`` skipped. 6500 is D65 — neutral daylight by
definition — which is what makes it the right zero, not merely the table's edge.
"""

MOST_TINT = 0.5
"""The far end of what ``tint`` may ask for — a gain swing of half a channel's light.

Around 0.1 the cast is plainly visible on white; at 0.5 a channel can arrive at half
or one-and-a-half strength, which is the outer edge of "a different shade of warm".
Past it the result reads as a broken sensor rather than a colour, so the constructor
refuses rather than lets a config typo produce one silently.
"""

_SPREADABLE = 2
"""Pixels a side below which there is no unevenness to draw.

A one-pixel image has no *inside* for warmth to vary across, and the vendor's plasma
generator does not survive being asked — measured, a ``(1, 1)`` target raises
``OverflowError`` out of a ``log2(0)``. Anything larger is fine, including shapes
that are neither square nor a power of two.
"""

_EQUALISER_BINS = 512
"""Resolution of the area-equalising step; see ``_by_area``.

Exact ranking costs 35 ms on a 512x512 field and 167 ms at 1024x1024 — real money in
a dataloader. A 512-bin histogram gets within 0.0009 of the exact rank, which at any
usable spread is under a kelvin, for less than half the time.
"""


class MaskedPlanckianJitter(A.CustomTransformsApplyMixin, A.ImageOnlyTransform):
    """Warm the image towards candlelight inside one mask, patchily, and report the mean.

    ``A.PlanckianJitter`` warms the whole frame to one temperature; this warms only
    where a mask says the object is, varies the warmth *within* that region, and writes
    the region's mean temperature into a bound label. So a run learns to read the warmth
    of a region rather than of a photograph — the rest of the frame stays as it was and
    cannot be used as a shortcut.

    Three things decide whether the patchiness is actually visible, and all three had
    to be got right before it was:

    - **The swing is spread by area, not by value.** A plasma field's values cluster:
      measured, 73% of a region's pixels land in the middle third of its own range, so
      stretching that range across the swing leaves almost the whole region at one
      intensity — which is exactly what it looked like. Ranking the values instead gives
      every intensity an equal share of the area, and the patches appear.
    - **The swing is measured inside the mask.** The generator spans ``[0, 1]`` over the
      whole *image*, so a small region catches only a sliver of it — measured, a mask
      covering 1% of the frame realised 33% of the spread it asked for. Equalising
      within the mask makes ``spread`` mean the same thing at any region size.
    - **``temperature_range`` bounds the label, not the pixels.** The swing runs past it
      and is clipped by the coefficient table instead. Bounding both with one number is
      what used to force a wide swing into the middle of the range, where the ratio is
      flat and nothing shows.

    **The label is the region's mean temperature, measured rather than assumed.** It is
    read off the finished field, so clipping at the table's edge cannot make it lie —
    a centre near 3000 K loses the cold half of its swing, and the label says so.

    The label carries **kelvin**, which is worth two sentences because both are easy to
    be surprised by. It runs *backwards* to the effect: 3000 is the most yellow and 6500
    the least. And ``ScalarTargetEncoder`` passes a continuous target through as
    ``float(value)``, so a regression head sees raw kelvin rather than something around
    zero — scale it in the encoder, not here, where the number would stop meaning what
    its name says.

    An **empty mask draws no warmth**: there is nothing to warm, so the image comes back
    untouched and the label reports the cool end of the range rather than a temperature
    nothing was warmed to. Labelling an unchanged image as warmed is the same defect
    ``RandomBorderCrop.min_crop`` exists to prevent, met from the other side — there, a
    crop too small to see; here, a region with no pixels in it.

    The mask itself is **not** modified: this is an ``ImageOnlyTransform``, so a spatial
    target reaches it as data and leaves as it arrived. Its edge stays hard — the warmth
    stops where the mask does. Feathering that boundary is a separate decision, and a
    real one for a mask that came from ground truth, where the step lines up with the
    object exactly.

    Bind the label with ``AlbumentationsTransform(label_targets=["warmth"])``, and carry
    the mask as an auxiliary input — ``data.auxiliary_inputs: {lesion: {column: mask_path}}``
    reaches the pipeline on its own, and never reaches the batch. A mask that is *also* a
    segmentation target arrives the same way through ``spatial_targets``; ``mask_key``
    reads either, and names which one this augmentation is about.

    Parameters:
        mask_key (str): Which of the sample's spatial targets bounds the warmth. Named
            rather than guessed: a sample may carry several masks, and only the
            experiment knows which one this augmentation is about.
        temperature_range (tuple[int, int]): Kelvin the region's **mean** is drawn from,
            inclusive at both ends. Individual pixels run past it by up to half the
            spread; the coefficient table is what bounds them.
        spread (int | tuple[int, int]): The full width, in kelvin, of the swing across
            the region — so a pixel sits at most half of it from the drawn centre. A
            single number is that width every time; a range draws a fresh width per
            application, so one sample is even and the next is mottled. ``0`` warms
            the region evenly, which is ``planckian_jitter`` exactly.
        tint (float): How far, as a fraction, each channel's gain may wander off the
            planckian locus — every application draws one gain per channel from
            ``[1 - tint, 1 + tint]`` and multiplies it into the warming. ``0.1`` is a
            visible cast; the locus itself only ever trades red against blue, so this
            is where any green-versus-magenta variety comes from. **The cast follows
            the warmth, per pixel**: at 3000 K a gain applies whole, halfway it
            applies by half, and at 6500 K it is gone — a cast is a property of the
            illumination, so where nothing warms, nothing tinges, and a cool sample
            stays the original. Multiplicative rather than an additive ``RGBShift``
            on purpose: gains compose with the planckian coefficients as one
            illumination and leave black pixels black, where an additive shift would
            tint pigment that reflects no light. The label does not report it — the
            tint is nuisance variation, there so the regression cannot pin its answer
            to one exact hue.
        roughness (float | tuple[float, float]): How fine the patches are, in
            ``[0, 1]``. Measured by the correlation between neighbouring pixels:
            ``0.1`` gives one broad gradient (0.99), ``0.5`` a few large patches
            (0.90), ``0.9`` a fine mottle (0.37). A single number is that texture
            every time; a range draws a fresh one per application, so one sample
            fades across and the next is mottled.
        mode (Mode): ``blackbody`` follows an ideal radiator, ``cied`` the CIE daylight
            series — which starts at 4000 K and so cannot reach the warmest end.
        p (float): Probability of warming at all. Below 1, the label keeps whatever the
            dataset gave it on the samples that are skipped, so that value has to
            already mean "not warmed".
    """

    def __init__(
        self,
        mask_key: str,
        temperature_range: tuple[int, int] = (WARMEST, COOLEST),
        spread: int | tuple[int, int] = 0,
        roughness: float | tuple[float, float] = 0.5,
        tint: float = 0.0,
        mode: Mode = "blackbody",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        low, high = temperature_range
        covered = _bounds(mode)
        if low > high:
            raise ValueError(f"temperature_range runs low to high, got ({low}, {high}).")
        if low < covered[0] or high > covered[1]:
            raise ValueError(
                f"'{mode}' is tabulated over {covered[0]}..{covered[1]} K, "
                f"and temperature_range asks for {low}..{high}."
            )
        # One number is a width used every time; a pair is the range a width is drawn
        # from per application. Held as a range either way, so one code path draws.
        listed = [spread, spread] if isinstance(spread, int) else [int(one) for one in spread]
        if len(listed) != 2 or listed[0] > listed[1]:
            raise ValueError(f"spread is a width or a (low, high) range of widths, got {spread}.")
        if listed[0] < 0:
            raise ValueError(f"spread is a width in kelvin and cannot be negative, got {spread}.")
        texture = [roughness, roughness] if isinstance(roughness, int | float) else [float(one) for one in roughness]
        if len(texture) != 2 or texture[0] > texture[1]:
            raise ValueError(f"roughness is a value or a (low, high) range of them, got {roughness}.")
        if not 0.0 <= texture[0] <= texture[1] <= 1.0:
            raise ValueError(f"roughness runs from 0 (one broad gradient) to 1 (a fine mottle), got {roughness}.")
        if not 0.0 <= tint <= MOST_TINT:
            raise ValueError(
                f"tint is a fractional gain swing per channel and runs 0..{MOST_TINT} — beyond that a "
                f"channel can lose most of its light, which is a colour failure, not a cast. Got {tint}."
            )
        self.mask_key = mask_key
        self.temperature_range = temperature_range
        self.spread = (listed[0], listed[1])
        self.roughness = (texture[0], texture[1])
        self.tint = tint
        self.mode = mode

    def get_params_dependent_on_data(self, params: dict[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        region = _region(data, self.mask_key, params["shape"])
        height, width = params["shape"][:2]
        if not region.any():
            # Nothing to warm, so nothing is drawn; see the class docstring.
            return {
                "temperature": float(self.temperature_range[1]),
                "field": np.empty(0),
                "region": region,
                "gains": (1.0, 1.0, 1.0),
            }
        centre = self.py_random.randint(*self.temperature_range)
        field = self._field(
            centre,
            self.py_random.randint(*self.spread),
            self.py_random.uniform(*self.roughness),
            region,
            (int(height), int(width)),
        )
        gains = tuple(self.py_random.uniform(1.0 - self.tint, 1.0 + self.tint) for _ in range(3))
        # Read back rather than reported as drawn: where the swing ran off the table it
        # was clipped, and the label has to be what the region actually is.
        return {"temperature": float(field.mean()), "field": field, "region": region, "gains": gains}

    def apply(
        self,
        img: np.ndarray,
        field: np.ndarray,
        region: np.ndarray,
        gains: tuple[float, float, float],
        **params: Any,
    ) -> np.ndarray:
        warmed = img.copy()
        warmed[region] = _warmed(img[region], field, self.mode, gains)
        return warmed

    def apply_to_label(self, label: Any, temperature: float, **params: Any) -> float:
        return int(temperature)

    def _field(
        self, centre: int, spread: int, roughness: float, region: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """One temperature per warmed pixel, in the order ``img[region]`` reads them."""
        if not spread or max(shape) < _SPREADABLE:
            return np.full(int(region.sum()), float(centre))
        plasma = generate_plasma_pattern(
            target_shape=shape,
            roughness=roughness,
            random_generator=self.random_generator,
        )[region]
        spanned = _by_area(plasma)
        return np.clip(centre + spread * (spanned - spanned.mean()), *_bounds(self.mode))


def _by_area(values: np.ndarray) -> np.ndarray:
    """Each value replaced by the share of the region that is no warmer, in ``[0, 1]``.

    Equalisation, and the reason the patches are visible at all. A plasma field's values
    cluster around their own middle — measured, 73% of a region's pixels sit in the
    middle third of its range at ``roughness=0.5`` — so mapping the range linearly onto
    the swing puts almost the whole region at one intensity and spends the extremes on a
    handful of pixels. Ranking gives every intensity the same share of the area, which
    is what "patchy" means to the eye.
    """
    counts, edges = np.histogram(values, bins=_EQUALISER_BINS, range=(0.0, 1.0))
    cumulative = np.cumsum(counts) / counts.sum()
    return np.asarray(np.interp(values, edges[1:], cumulative))


def _bounds(mode: Mode) -> tuple[int, int]:
    """The coldest and warmest temperature this mode is tabulated for."""
    covered = PLANCKIAN_COEFFS[mode]
    return min(covered), max(covered)


def _warmed(
    pixels: np.ndarray, field: np.ndarray, mode: Mode, gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """The warmed pixels, each at its own temperature; ``pixels`` is ``[N, 3]``.

    The coefficients come from the table ``planckian_jitter`` itself reads, and are
    interpolated as whole ``[r, g, b]`` triples then divided by green *afterwards* —
    which is the order the vendor uses. Interpolating the two ratios directly is the
    intuitive reading and the wrong one: measured, it drifts up to 0.005 in the blue
    multiplier, and blue is where nearly all the yellowing lives.

    Two deliberate departures from ``planckian_jitter``, both anchored at ``COOLEST``:

    - **The multipliers are normalised to their value at 6500 K**, so a pixel whose
      field says neutral is the original byte for byte — see ``COOLEST`` for why the
      vendor's own table would brighten it by 4% instead, and what that teaches a model.
    - **The tint fades with the warmth, per pixel**: each gain is pulled towards one by
      how close that pixel's temperature is to neutral. A cast is a property of the
      illumination, and a lamp too weak to yellow is too weak to tinge — without this,
      a ±15% gain would dwarf the 8% shift that is all a 6000 K pixel gets.

    The cast truncates, as the vendor's uint8 lookup table does — measured, rounding
    instead disagrees on 29% of channels by one grey level — while on a float image
    the same expression is no cast at all.

    Only the warmed pixels are passed in, so the cost follows the region rather than the
    frame: a lesion covering a twentieth of a 512×512 image does a twentieth of the work.
    """
    table = PLANCKIAN_COEFFS[mode]
    temperatures = np.array(sorted(table), dtype=np.float64)
    triples = np.array([table[int(one)] for one in temperatures], dtype=np.float64)
    red, green, blue = (np.interp(field, temperatures, triples[:, channel]) for channel in range(3))
    neutral_red, neutral_green, neutral_blue = (
        np.interp(COOLEST, temperatures, triples[:, channel]) for channel in range(3)
    )
    warmth = np.clip((COOLEST - field) / (COOLEST - WARMEST), 0.0, 1.0)
    warmed = pixels.astype(np.float64)
    warmed[:, 0] *= (1.0 + (gains[0] - 1.0) * warmth) * (red / green) / (neutral_red / neutral_green)
    warmed[:, 1] *= 1.0 + (gains[1] - 1.0) * warmth
    warmed[:, 2] *= (1.0 + (gains[2] - 1.0) * warmth) * (blue / green) / (neutral_blue / neutral_green)
    return np.asarray(np.clip(warmed, 0, MAX_VALUES_BY_DTYPE[pixels.dtype]), dtype=pixels.dtype)


def _region(data: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """The mask as a plain ``[H, W]`` boolean, whatever shape it arrived in.

    Albumentations hands a mask on with a channel axis it did not have going in —
    measured, a ``[12, 16]`` mask reads as ``[12, 16, 1]`` here — so the axis is
    reduced rather than assumed away. Any positive value counts as inside.
    """
    if key not in data:
        offered = ", ".join(f"'{name}'" for name in sorted(data)) or "nothing"
        raise KeyError(
            f"MaskedPlanckianJitter bounds its warmth by '{key}', which this pipeline was not given. "
            f"Declare it under data.auxiliary_inputs (or, for a mask that is also a task's target, "
            f"AlbumentationsTransform(spatial_targets=['{key}'])). It was handed {offered}."
        )
    region = np.asarray(data[key]) > 0
    if region.ndim > 2:
        region = region.any(axis=tuple(range(2, region.ndim)))
    if region.shape != tuple(shape[:2]):
        raise ValueError(
            f"MaskedPlanckianJitter cannot warm '{key}' onto the image: the mask is {region.shape} "
            f"and the image {tuple(shape[:2])}. They must be at the same resolution."
        )
    return region
