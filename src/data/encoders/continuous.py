"""Numeric targets: a scalar as it is, or as a distribution over bins."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, ClassVar, override

import numpy as np

from src.core.entities import Distribution
from src.data.encoders.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import measured

log = logging.getLogger(__name__)


@target_encoder_registry.register("scalar")
class ScalarTargetEncoder(TargetEncoder):
    """Real-valued targets; nothing to fit."""

    def encode(self, value: Any) -> float:
        return float(value)

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        return measured(values)


class BinnedTargetEncoder(TargetEncoder):
    """Base for encoders spreading one continuous value over ordered bins.

    Label distribution learning: the model expresses "around 42" and is read back as the
    distribution's expectation. The range is learned from the training split unless declared
    and padded (``_padding``), because bins ending at the data's extremes clip edge values
    and the expectation no longer equals the value. ``num_classes`` is the bin count, so a
    head sizes itself as for any categorical task; ``class_values`` carries the bin centres.

    Parameters:
        bins (int): Number of bins the value is spread over.
        low (float | None): Smallest value to represent; learned when omitted.
        high (float | None): Largest value to represent; learned when omitted.
    """

    MINIMUM_BINS: ClassVar[int] = 2
    """Fewer than two bins is not a distribution over anything.

    Named because a subclass with a stricter rule of its own has to know where the plain
    one starts, and speak only above it — a particular refusal over a basic mistake
    tells the user about the wrong thing.
    """

    def __init__(self, bins: int = 20, low: float | None = None, high: float | None = None) -> None:
        if bins < self.MINIMUM_BINS:
            raise ValueError(f"{type(self).__name__} needs at least {self.MINIMUM_BINS} bins, got {bins}.")
        if (low is None) != (high is None):
            raise ValueError(f"{type(self).__name__} takes both 'low' and 'high' or neither, not one.")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"{type(self).__name__} needs low < high, got low={low}, high={high}.")
        self._bins = bins
        self._declared = low is not None
        self._centers: np.ndarray | None = None
        if low is not None and high is not None:
            self._lay_out_bins(low, high)

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """The column as it arrives, not as the bins encode it.

        A dataset report answers "what am I training on", and the answer is the
        continuous target the user wrote down — the binning is this encoder's
        choice about how to learn it, not a fact about the data.
        """
        return measured(values)

    def fit(self, values: Iterable[Any]) -> None:
        if self._declared:
            return
        numbers = np.asarray([float(value) for value in values], dtype=np.float64)
        if numbers.size == 0:
            raise ValueError(f"{type(self).__name__} cannot learn a range from an empty training split.")
        low, high = float(numbers.min()), float(numbers.max())
        if low == high:
            raise ValueError(f"{type(self).__name__} cannot bin a constant target: every training value is {low}.")
        self._lay_out_bins(low, high)

    def _lay_out_bins(self, low: float, high: float) -> None:
        """Place the bin centres, which are the midpoints of evenly spaced edges."""
        padding = self._padding(low, high)
        edges = np.linspace(low - padding, high + padding, self._bins + 1, dtype=np.float64)
        self._centers = (edges[:-1] + edges[1:]) / 2.0

    def _padding(self, low: float, high: float) -> float:
        """Room to leave beyond the data on each side, in target units."""
        return 0.0

    def _require_centers(self) -> np.ndarray:
        if self._centers is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(train_values) first.")
        return self._centers

    @property
    def bin_width(self) -> float:
        """Distance between neighbouring bin centres, once the range is known."""
        centers = self._require_centers()
        return float(centers[1] - centers[0])

    @property
    def num_classes(self) -> int | None:
        return self._bins if self._centers is not None else None

    @property
    def class_values(self) -> list[float] | None:
        return [float(center) for center in self._centers] if self._centers is not None else None


@target_encoder_registry.register("gaussian_bins")
class GaussianBinsTargetEncoder(BinnedTargetEncoder):
    """A continuous value as a normal density over bin centres, normalised to sum 1.

    Neighbouring bins carry real mass, so near-misses are taught as nearly right. ``sigma``
    defaults to one bin width; much smaller collapses the density into one bin and turns the
    scheme into plain classification. The range is padded by three sigma so the density is
    never clipped.

    Parameters:
        bins (int): Number of bins the value is spread over.
        sigma (float | None): Width of the smoothing, in target units; defaults to one bin width.
        low (float | None): Smallest value to represent; learned when omitted.
        high (float | None): Largest value to represent; learned when omitted.
    """

    NARROW_SIGMA_RATIO: ClassVar[float] = 0.35
    """Below this sigma-to-bin-width ratio the density is one-hot in all but name."""

    SIGMAS_OF_ROOM: ClassVar[float] = 3.0
    """How many sigma of room the range is padded by on each side, so nothing is clipped.

    Three is where a normal density is spent: beyond it lies 0.3% of the mass, which is
    below what the expectation this encoding is read back by can notice.
    """

    def __init__(
        self,
        bins: int = 20,
        sigma: float | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        if sigma is not None and sigma <= 0:
            raise ValueError(f"gaussian_bins needs a positive sigma, got {sigma}.")
        # Above the base's own minimum only: `bins < 2` is the plainer mistake and the base
        # names it, so this more particular rule must not speak over it.
        if sigma is None and self.MINIMUM_BINS <= bins <= 2 * self.SIGMAS_OF_ROOM:
            # An undeclared sigma *is* the bin width, and the padding widens the bins it
            # is measured against — so the two are one equation, and below this many bins
            # it has no solution: the room needed grows without bound.
            raise ValueError(
                f"gaussian_bins takes its sigma from the bin width and pads the range by "
                f"{self.SIGMAS_OF_ROOM:g} of them, which {bins} bins cannot both satisfy. "
                f"Declare a 'sigma', or use at least {int(2 * self.SIGMAS_OF_ROOM) + 1} bins."
            )
        self._declared_sigma = sigma
        self._sigma = 0.0  # Resolved with the bins, from the declaration or the bin width.
        super().__init__(bins=bins, low=low, high=high)

    @override
    def _padding(self, low: float, high: float) -> float:
        """Three sigma of room on each side, so an edge value's density is never clipped.

        An undeclared sigma *is* the bin width, which this padding changes, so the two are solved
        together: with ``width = (span + 2·room)/bins`` and ``room = 3·width``, ``room =
        3·span/(bins - 6)``. Measured over a span of 100, the approximation ``3·span/(bins - 1)``
        leaves 2.00 sigma at 10 bins and drifts the lowest value's expectation inward by 0.797.
        """
        declared = self._declared_sigma
        if declared is not None:
            return self.SIGMAS_OF_ROOM * declared
        return self.SIGMAS_OF_ROOM * (high - low) / (self._bins - 2 * self.SIGMAS_OF_ROOM)

    @override
    def _lay_out_bins(self, low: float, high: float) -> None:
        super()._lay_out_bins(low, high)
        if self._declared_sigma is None:
            self._sigma = self.bin_width
            return
        self._sigma = self._declared_sigma
        if self._sigma < self.NARROW_SIGMA_RATIO * self.bin_width:
            log.warning(
                "gaussian_bins sigma %.4g is small next to the %.4g bin width: the density lands "
                "almost entirely in one bin, which is plain classification over quantised values. "
                "Raise sigma towards the bin width, or use fewer bins.",
                self._sigma,
                self.bin_width,
            )

    def encode(self, value: Any) -> np.ndarray:
        centers = self._require_centers()
        density = np.exp(-0.5 * ((centers - float(value)) / self._sigma) ** 2)
        total = density.sum()
        if total <= 0.0:
            # Far outside the fitted range (val and test are not bound by it): all the
            # mass the value deserves belongs to the nearest bin rather than nowhere.
            density = np.zeros_like(centers)
            density[int(np.abs(centers - float(value)).argmin())] = 1.0
            total = 1.0
        normalized: np.ndarray = (density / total).astype(np.float32)
        return normalized


@target_encoder_registry.register("linear_bins")
class LinearBinsTargetEncoder(BinnedTargetEncoder):
    """A continuous value split between the two bin centres that bracket it.

    The encoding behind Distribution Focal Loss: two bins carry mass in linear proportion,
    so the expectation reproduces the value exactly inside the centre range — exactness for
    no smooth mass around the value. The range is padded by half a bin so edge values sit on
    centres rather than clamp.

    Parameters:
        bins (int): Number of bins the value is spread over.
        low (float | None): Smallest value to represent; learned when omitted.
        high (float | None): Largest value to represent; learned when omitted.
    """

    @override
    def _padding(self, low: float, high: float) -> float:
        # Half a bin: it puts the first and last centres exactly on low and high.
        return (high - low) / (2.0 * (self._bins - 1))

    def encode(self, value: Any) -> np.ndarray:
        centers = self._require_centers()
        distribution = np.zeros(centers.size, dtype=np.float32)
        clamped = float(np.clip(float(value), centers[0], centers[-1]))
        upper = int(np.searchsorted(centers, clamped, side="left"))
        if centers[upper] == clamped:
            distribution[upper] = 1.0
            return distribution
        gap = centers[upper] - centers[upper - 1]
        distribution[upper - 1] = (centers[upper] - clamped) / gap
        distribution[upper] = (clamped - centers[upper - 1]) / gap
        return distribution
