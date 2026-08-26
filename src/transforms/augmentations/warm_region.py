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

    ``A.PlanckianJitter`` warms the whole frame; this warms only where a mask says the
    object is, varies the warmth within the region, and writes the region's mean temperature
    (kelvin; 3000 is the most yellow) into a bound label. An empty mask warms nothing and the
    label says the cool end. Bind the label with
    ``AlbumentationsTransform(label_targets=["warmth"])`` and carry the mask as an auxiliary input.

    Parameters:
        mask_key (str): Which of the sample's mask arrays bounds the warmth.
        temperature_range (tuple[int, int]): Kelvin the region's mean is drawn from, inclusive.
        spread (int | tuple[int, int]): Full width, in kelvin, of the swing across the region;
            a range draws a fresh width per application. ``0`` warms evenly.
        tint (float): How far each channel's gain may wander off the planckian locus; the
            cast follows the warmth per pixel and is not reported in the label.
        roughness (float | tuple[float, float]): How fine the patches are, in ``[0, 1]``.
            Measured by neighbour correlation: ``0.1`` one broad gradient, ``0.9`` fine mottle.
        mode (Mode): ``blackbody`` or ``cied`` (starts at 4000 K, cannot reach the warmest end).
        p (float): Probability of warming at all; skipped samples keep the dataset's label.
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

    Coefficients are interpolated as ``[r, g, b]`` triples then divided by green, the
    vendor's order — measured, interpolating the ratios directly drifts the blue multiplier
    by 0.005. Two departures from ``planckian_jitter``, both anchored at ``COOLEST``: the
    multipliers are normalised to 6500 K so a neutral pixel is the original byte for byte,
    and the tint fades with the warmth per pixel. The cast truncates as the vendor's uint8
    table does. Only the warmed pixels are passed in, so the cost follows the region.
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
            f"AlbumentationsTransform(targets={{'{key}': Geometry.MASK}})). It was handed {offered}."
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
