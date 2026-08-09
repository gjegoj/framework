"""Target encoders: raw column values into the values a task's target starts as."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np

from src.core.entities import ClassDistribution, Distribution, TargetFacts
from src.core.vocabulary import ordered_names
from src.data.cache import cached
from src.data.loaders import ImageLoader
from src.data.registry import target_encoder_registry
from src.data.statistics import counted, measured

if TYPE_CHECKING:
    from src.data.cache import LoaderCache
    from src.data.loaders import InputLoader

log = logging.getLogger(__name__)


class TargetEncoder(ABC):
    """Encodes one task's raw target values.

    An encoder stays on the *raw* side of the pipeline — a label becomes a class index,
    a mask becomes an array — because a transform may still have to touch the result: a
    mask follows the image's geometry. Tensors are made exactly once afterwards, by the
    transform ending in ``ToTensorV2`` or by collation.

    ``fit`` learns vocabulary or statistics from the training split and is a no-op by
    default. Afterwards the encoder exposes what it inferred (``num_classes``,
    ``class_names``), which the ``DataModule`` records into the ``DataProfile``.

    ``spatial`` marks encoders whose values live in image space: a transform must carry
    those through the same geometry as the image, and only the encoder knows it.
    """

    spatial: ClassVar[bool] = False

    def fit(self, values: Iterable[Any]) -> None:
        """Learn from training-split values. Default: nothing to learn."""

    @abstractmethod
    def encode(self, value: Any) -> Any:
        """Encode one raw value into the target's pre-tensor form."""

    @property
    def num_classes(self) -> int | None:
        """Label-vocabulary size, ``None`` for class-free targets."""
        return None

    @property
    def class_names(self) -> list[str] | None:
        """Class names aligned with encoded indices, ``None`` when class-free."""
        return None

    @property
    def class_values(self) -> list[float] | None:
        """The number each encoded position stands for, ``None`` when unordered.

        Set by encoders that spread one continuous value over ordered classes:
        the values are what turns a predicted distribution back into a number.
        """
        return None

    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """What this column holds, or ``None`` when this encoder does not describe it.

        Beside ``facts()``, and for the same reason: the encoder owns the
        vocabulary and the parsing, so nothing else can count its own column
        correctly. A method on the base class rather than an optional capability
        bolted on — an encoder that says nothing returns ``None`` and the report
        names the task anyway, where a missing method dropped the column in
        silence and left the reader to guess which of their targets was gone.
        """
        return None

    def facts(self) -> TargetFacts:
        """What fitting this encoder inferred, as the one record a profile stores.

        Reporting the facts together is what keeps a caller from enumerating
        them: a new kind of fact is then declared by the encoders that have it,
        not by everything that fills a ``DataProfile``.
        """
        return TargetFacts(
            num_classes=self.num_classes,
            class_names=self.class_names,
            class_values=self.class_values,
        )


@target_encoder_registry.register("label")
class LabelTargetEncoder(TargetEncoder):
    """Categorical labels into class indices.

    A declared vocabulary is the contract the data is validated against —
    a typo row fails loudly instead of silently growing the class count, and
    the index space stays put when a resample drops a rare class from train.
    Undeclared, the vocabulary is learned from the training split, sorted.

    Parameters:
        classes (Mapping[int, str] | None): Declared vocabulary, index to name.
    """

    def __init__(self, classes: Mapping[int, str] | None = None) -> None:
        names = ordered_names(classes) if classes is not None else None
        self._declared = names is not None
        self._index: dict[str, int] | None = (
            {name: position for position, name in enumerate(names)} if names is not None else None
        )

    def fit(self, values: Iterable[Any]) -> None:
        if self._declared:
            assert self._index is not None
            unknown = sorted({str(value) for value in values} - self._index.keys())
            if unknown:
                known = ", ".join(self._index)
                raise LookupError(f"Values outside the declared classes: {', '.join(unknown)}. Declared: {known}.")
            return
        vocabulary = sorted({str(value) for value in values})
        self._index = {name: position for position, name in enumerate(vocabulary)}

    def encode(self, value: Any) -> int:
        if self._index is None:
            raise RuntimeError("LabelTargetEncoder is not fitted; call fit(train_values) first.")
        try:
            return self._index[str(value)]
        except KeyError:
            known = ", ".join(self._index)
            raise LookupError(f"Unknown label '{value}'. Known classes: {known}.") from None

    @property
    def num_classes(self) -> int | None:
        return len(self._index) if self._index is not None else None

    @property
    def class_names(self) -> list[str] | None:
        return list(self._index) if self._index is not None else None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per row, seeded with the vocabulary so an unused class still shows."""
        return counted(self.class_names, (str(value) for value in values))


@target_encoder_registry.register("multilabel")
class MultiLabelTargetEncoder(TargetEncoder):
    """Several labels per row into one indicator vector; the vocabulary is learned at fit.

    Cells hold either a separated string (``"cat,dog"``) or a real list, the
    form JSON tables arrive in. A row with no labels encodes to all zeros
    rather than counting as missing: in multi-label the absence of every class
    is itself a valid observation.

    Values are ``float`` because binary cross-entropy compares against
    probabilities — encoding to ``int`` would only push a cast downstream.

    Parameters:
        classes (Mapping[int, str] | None): Declared vocabulary, index to name —
            the contract the data is validated against. ``None`` learns it from
            the training split, sorted.
        separator (str): Separator splitting a string cell into labels.
    """

    def __init__(self, classes: Mapping[int, str] | None = None, separator: str = ",") -> None:
        if not separator:
            raise ValueError("MultiLabelTargetEncoder needs a non-empty separator.")
        self._separator = separator
        self._declared = classes is not None
        self._classes: list[str] | None = None
        self._positions: dict[str, int] = {}
        if classes is not None:
            self._adopt(ordered_names(classes))

    def _adopt(self, names: list[str]) -> None:
        """Hold the vocabulary and the index into it together, so the two cannot drift.

        The index used to be rebuilt inside ``encode`` — once per row, every epoch, for a
        mapping that is fixed the moment the vocabulary is. It is derived here instead,
        at both of the two points a vocabulary arrives: declared, or learned at ``fit``.
        """
        self._classes = names
        self._positions = {name: position for position, name in enumerate(names)}

    def fit(self, values: Iterable[Any]) -> None:
        if self._declared:
            assert self._classes is not None
            declared = set(self._classes)
            unknown = sorted({label for value in values for label in self._labels_in(value)} - declared)
            if unknown:
                known = ", ".join(self._classes)
                raise LookupError(f"Labels outside the declared classes: {', '.join(unknown)}. Declared: {known}.")
            return
        self._adopt(sorted({label for value in values for label in self._labels_in(value)}))

    def encode(self, value: Any) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("MultiLabelTargetEncoder is not fitted; call fit(train_values) first.")
        indicator = np.zeros(len(self._classes), dtype=np.float32)
        for label in self._labels_in(value):
            try:
                indicator[self._positions[label]] = 1.0
            except KeyError:
                known = ", ".join(self._classes)
                raise LookupError(f"Unknown label '{label}'. Known classes: {known}.") from None
        return indicator

    def _labels_in(self, value: Any) -> set[str]:
        """The labels a cell carries, in either of the two forms a table stores them."""
        if isinstance(value, list | tuple | set):
            return {str(item).strip() for item in value if str(item).strip()}
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return set()
        return {part.strip() for part in str(value).split(self._separator) if part.strip()}

    @property
    def num_classes(self) -> int | None:
        return len(self._classes) if self._classes is not None else None

    @property
    def class_names(self) -> list[str] | None:
        return list(self._classes) if self._classes is not None else None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """One count per label, so the total exceeds the row count wherever rows carry several."""
        return counted(self.class_names, (label for value in values for label in self._labels_in(value)))


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

    Learning a continuous target as a distribution over bins (label distribution
    learning) lets a model express uncertainty — "around 42, and 41 or 43 would
    not be wrong" — and is read back as the distribution's expectation.

    The range is learned from the training split unless declared, which is what
    keeps the encoding faithful: bins ending exactly at the data's extremes clip
    the mass of values sitting on the edge, and the expectation of a clipped
    distribution no longer equals the value it encoded. Subclasses say through
    ``_padding`` how much room beyond the data their scheme needs.

    ``num_classes`` is the bin count, so a head sizes itself the same way it does
    for any categorical task, and ``class_values`` carries the bin centres on to
    whatever has to turn a prediction back into a number.

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

    The smoothing is the point: neighbouring bins carry real mass, so the model
    is taught that near-misses are nearly right. ``sigma`` defaults to one bin
    width, the scale at which that actually happens — a much smaller sigma
    collapses the density into a single bin and quietly turns the whole scheme
    into plain classification over quantised values.

    The range is padded by three sigma so the density is never clipped, which is
    what keeps the encoded distribution's expectation equal to the value itself.

    Parameters:
        bins (int): Number of bins the value is spread over.
        sigma (float | None): Width of the smoothing, in target units;
            defaults to one bin width.
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

        A declared sigma is a length, so the room is three of it. An undeclared one *is*
        the bin width — and the bin width is what this padding changes — so the two are
        solved together rather than approximated: with ``width = (span + 2·room)/bins``
        and ``room = 3·width``, the room is ``3·span/(bins - 6)``.

        Approximating it by ``3·span/(bins - 1)`` — the spacing the centres would have
        if nothing were padded — falls short at every bin count, and that shortfall is
        what an edge value pays. Measured over a span of 100: 2.00 sigma of room at 10
        bins, 2.40 at 20, 2.73 at 50, drifting the expectation of the lowest value
        inward by 0.797, 0.120 and 0.016 respectively.
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

    The encoding behind Distribution Focal Loss: exactly two bins carry mass, in
    linear proportion, so the distribution's expectation reproduces the value
    exactly anywhere inside the centre range. The trade against ``gaussian_bins``
    is an exact expectation for no smooth mass around the value.

    The range is padded by half a bin so the outermost data values sit on centres
    rather than beyond them, where they would clamp.

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


@target_encoder_registry.register("mask")
class MaskTargetEncoder(TargetEncoder):
    """Segmentation masks: an image file of class indices into an ``[H, W]`` array.

    Reading is delegated to a grayscale ``ImageLoader``, so mask files get the
    same root handling and the same diagnostics as image inputs. The class
    count cannot be inferred without reading every mask, so it is declared —
    as a bare ``num_classes``, or through the task's ``classes`` vocabulary,
    which also gives the classes their names.

    Parameters:
        num_classes (int | None): Number of segmentation classes, background
            included; derived from ``classes`` when a vocabulary is declared.
        classes (Mapping[int, str] | None): Declared vocabulary, index to name.
        root (str | Path | None): Prefix for the mask paths stored in the table.
        cache (LoaderCache | None): Serves mask reads from memory when given;
            assembly offers one as a derived value.
    """

    spatial: ClassVar[bool] = True

    def __init__(
        self,
        num_classes: int | None = None,
        classes: Mapping[int, str] | None = None,
        root: str | Path | None = None,
        cache: LoaderCache | None = None,
    ) -> None:
        names = ordered_names(classes) if classes is not None else None
        if names is None and num_classes is None:
            raise ValueError("A mask needs its class count: declare 'num_classes' or 'classes'.")
        if names is not None and num_classes is not None and num_classes != len(names):
            raise ValueError(f"num_classes={num_classes} disagrees with {len(names)} declared classes; declare one.")
        resolved = num_classes if num_classes is not None else len(names or ())
        if resolved < 1:
            raise ValueError(f"num_classes must be positive, got {resolved}.")
        self._num_classes = resolved
        self._names = names
        # The mask is read through a loader this encoder owns, so caching has to be
        # handed in: there is nothing on the outside left to wrap.
        read: InputLoader = ImageLoader(root=root, grayscale=True)
        self._read = cached(read, cache) if cache is not None else read

    def encode(self, value: Any) -> np.ndarray:
        mask: np.ndarray = self._read(value).astype(np.int64)
        return mask

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """Pixels per class, read from every mask — the class imbalance a loss will fight.

        Counted in full rather than sampled. Measured: 0.88 ms to decode a mask and bin
        its pixels, so 3.3 s for a 3680-mask dataset and about 18 s for 20,000 — once,
        before the first epoch. With a cache configured the reads are the same ones
        training is about to warm, so the pass costs nothing at all.

        A pixel holding an index the declared vocabulary does not reach is refused
        here rather than at the loss, where it surfaces as a shape error a thousand
        steps in.
        """
        names = self.class_names or [f"class{index}" for index in range(self._num_classes)]
        totals = np.zeros(self._num_classes, dtype=np.int64)
        for value in values:
            counts = np.bincount(self.encode(value).reshape(-1), minlength=self._num_classes)
            if counts.size > self._num_classes:
                raise ValueError(
                    f"Mask '{value}' holds class index {counts.size - 1}, but this task declares "
                    f"{self._num_classes} classes (0..{self._num_classes - 1}). Declare the missing "
                    f"classes, or remap the mask."
                )
            totals += counts
        return ClassDistribution(counts={name: int(total) for name, total in zip(names, totals, strict=True)})

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        return list(self._names) if self._names is not None else None
