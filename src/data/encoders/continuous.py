"""Numbers: as they are, or as a distribution over bins whose centres the task reads back."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable
from typing import ClassVar, Self

import torch
from torch import Tensor

from src.core import Distribution, TargetInfo
from src.data.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import measured

log = logging.getLogger(__name__)


class NumericEncoder(TargetEncoder):
    """What the encoders below share: the cells hold numbers, whatever is made of them afterwards.

    A binned target declares a vocabulary of bin centres, and its *column* is still a spread — which
    is why the shape of a report follows from the cells rather than from the facts an encoder settles.
    """

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        return measured(values)


@target_encoder_registry.register("scalar")
class ScalarEncoder(NumericEncoder):
    @property
    def info(self) -> TargetInfo:
        return TargetInfo()

    def encode(self, value: object) -> Tensor:
        return torch.tensor(float(value), dtype=torch.float32)  # type: ignore[arg-type]


class BinnedEncoder(NumericEncoder):
    """A number as a distribution over ``bins`` centres laid out over a range: declared, or learned on fit."""

    MINIMUM_BINS: ClassVar[int] = 2

    def __init__(self, bins: int = 20, low: float | None = None, high: float | None = None) -> None:
        if bins < self.MINIMUM_BINS:
            raise ValueError(f"{type(self).__name__} needs at least {self.MINIMUM_BINS} bins, got {bins}.")
        if (low is None) != (high is None):
            raise ValueError(f"{type(self).__name__} takes both low and high or neither.")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"{type(self).__name__} needs low < high, got {low} and {high}.")
        self.bins = bins
        self._declared = low is not None
        self._centers: Tensor | None = None
        if low is not None and high is not None:
            self._lay_out(low, high)

    @property
    def info(self) -> TargetInfo:
        centers = self._require_centers()
        values = tuple(float(center) for center in centers)
        return TargetInfo(classes={index: f"{value:g}" for index, value in enumerate(values)}, values=values)

    def fit(self, values: Iterable[object]) -> Self:
        if self._declared:
            return self
        numbers = torch.tensor([float(value) for value in values], dtype=torch.float64)  # type: ignore[arg-type]
        if numbers.numel() == 0:
            raise ValueError(f"{type(self).__name__} cannot learn a range from an empty training split.")
        low, high = float(numbers.min()), float(numbers.max())
        if low == high:
            raise ValueError(f"{type(self).__name__} cannot bin a constant target: every training value is {low}.")
        log.info(
            "%s learned its range from the training split: [%g, %g]; declare low and high to pin it.",
            type(self).__name__,
            low,
            high,
        )
        self._lay_out(low, high)
        return self

    def _lay_out(self, low: float, high: float) -> None:
        padding = self._padding(low, high)
        edges = torch.linspace(low - padding, high + padding, self.bins + 1, dtype=torch.float64)
        self._centers = (edges[:-1] + edges[1:]) / 2.0

    @abstractmethod
    def _padding(self, low: float, high: float) -> float:
        """How far past the observed range the outermost centres sit; every layout answers for itself."""

    def _require_centers(self) -> Tensor:
        if self._centers is None:
            raise ValueError(
                f"{type(self).__name__} is not fitted: fit it on the training split or declare low and high."
            )
        return self._centers

    @property
    def bin_width(self) -> float:
        centers = self._require_centers()
        return float(centers[1] - centers[0])


@target_encoder_registry.register("linear_bins")
class LinearBinsEncoder(BinnedEncoder):
    """Mass split between the two nearest centres in proportion to distance; the expectation returns the value."""

    def _padding(self, low: float, high: float) -> float:
        return (high - low) / (2.0 * (self.bins - 1))

    def encode(self, value: object) -> Tensor:
        centers = self._require_centers()
        distribution = torch.zeros(centers.numel(), dtype=torch.float32)
        clamped = min(max(float(value), float(centers[0])), float(centers[-1]))  # type: ignore[arg-type]
        upper = int(torch.searchsorted(centers, torch.tensor(clamped, dtype=torch.float64)))
        if float(centers[upper]) == clamped:
            distribution[upper] = 1.0
            return distribution
        gap = float(centers[upper] - centers[upper - 1])
        distribution[upper - 1] = (float(centers[upper]) - clamped) / gap
        distribution[upper] = (clamped - float(centers[upper - 1])) / gap
        return distribution


@target_encoder_registry.register("gaussian_bins")
class GaussianBinsEncoder(BinnedEncoder):
    """A Gaussian around the value, sampled at the centres; ``sigma`` defaults to the bin width."""

    NARROW_SIGMA_RATIO: ClassVar[float] = 0.35
    SIGMAS_OF_ROOM: ClassVar[float] = 3.0

    def __init__(
        self, bins: int = 20, sigma: float | None = None, low: float | None = None, high: float | None = None
    ) -> None:
        if sigma is not None and sigma <= 0:
            raise ValueError(f"gaussian_bins needs a positive sigma, got {sigma}.")
        if sigma is None and bins <= 2 * self.SIGMAS_OF_ROOM:
            raise ValueError(
                f"gaussian_bins takes its sigma from the bin width and pads the range by {self.SIGMAS_OF_ROOM:g} "
                f"of them, which {bins} bins cannot satisfy: declare sigma or use at least "
                f"{int(2 * self.SIGMAS_OF_ROOM) + 1} bins."
            )
        self._declared_sigma = sigma
        self.sigma = 0.0
        super().__init__(bins=bins, low=low, high=high)

    def _padding(self, low: float, high: float) -> float:
        if self._declared_sigma is not None:
            return self.SIGMAS_OF_ROOM * self._declared_sigma
        return self.SIGMAS_OF_ROOM * (high - low) / (self.bins - 2 * self.SIGMAS_OF_ROOM)

    def _lay_out(self, low: float, high: float) -> None:
        super()._lay_out(low, high)
        self.sigma = self._declared_sigma if self._declared_sigma is not None else self.bin_width
        if self.sigma < self.NARROW_SIGMA_RATIO * self.bin_width:
            log.warning(
                "gaussian_bins sigma %.4g is small next to the %.4g bin width: nearly all mass lands in one bin.",
                self.sigma,
                self.bin_width,
            )

    def encode(self, value: object) -> Tensor:
        centers = self._require_centers()
        density = torch.exp(-0.5 * ((centers - float(value)) / self.sigma) ** 2)  # type: ignore[arg-type]
        total = float(density.sum())
        if total <= 0.0:
            density = torch.zeros_like(centers)
            density[int((centers - float(value)).abs().argmin())] = 1.0  # type: ignore[arg-type]
            total = 1.0
        return (density / total).to(torch.float32)
