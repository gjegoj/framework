"""Data contracts: encoders for inputs and targets, the preprocessor that runs them, and the module that owns splits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ClassVar, Self

import pandas as pd
from torch.utils.data import Dataset

from src.core import (
    Batch,
    DatasetInfo,
    DatasetStatistics,
    Distribution,
    Geometry,
    InputInfo,
    Role,
    Sample,
    TargetInfo,
    TensorTree,
)
from src.transforms import SampleTransform


class Encoder(ABC):
    """Raw cell → training value, in two halves around augmentation.

    ``load`` prepares a raw cell for sample transforms (a path becomes pixels); ``encode``
    turns the transformed value into its tensor. Vocabularies are declared, never discovered:
    an encoder that reads one says so with ``takes_classes`` and receives ``classes`` from the
    task that declares them. The one exception is ``IdentityEncoder``, and what makes it one is
    that its vocabulary is never published — no output position is indexed by it, and nothing a
    deployment reads carries it; the reason is written where that encoder is.
    """

    geometry: ClassVar[Geometry] = Geometry.NONE
    takes_classes: ClassVar[bool] = False

    def load(self, value: object) -> object:
        return value

    def cache_key(self, value: object) -> str | None:
        """What identifies a loaded value across runs and roots; None means this load is not cached."""
        return None

    @abstractmethod
    def encode(self, value: object) -> TensorTree:
        raise NotImplementedError


class InputEncoder(Encoder):
    @property
    @abstractmethod
    def info(self) -> InputInfo:
        """Shape and modality of the encoded value, for heads, export examples and pages."""


class TargetEncoder(Encoder):
    """An encoder for what a run is scored against, which is the half of them a split is read for.

    ``fit`` and ``validate`` are here rather than on ``Encoder`` because only targets are ever given
    to them: the preprocessor fits the training split's targets and validates the other splits', and
    an input is only ever encoded. Offering them to every encoder invited an image encoder to
    implement a ``fit`` that would never be called.
    """

    @property
    @abstractmethod
    def info(self) -> TargetInfo:
        """Resolved target facts; available after ``fit`` for encoders that learn a layout."""

    def fit(self, values: Iterable[object]) -> Self:
        """Learn whatever layout this split settles — a range, a set of bins; a vocabulary is declared."""
        return self

    def validate(self, values: Iterable[object]) -> None:
        """Check a split against what was already learned, without learning from it.

        One encoder reads this differently and is allowed to: reading a split may widen what this can
        ``encode``, and may never widen what ``info`` publishes. That line is what keeps whatever is
        sized from a target — a head, an objective, a metric — sized by the training split alone.
        """
        return None

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        """What one split's raw cells of this column look like, for the report a run can print.

        Here rather than worked out from ``info``, because what a cell *holds* is what decides the
        shape — a binned target declares a vocabulary and still holds numbers, and a mask holds a path
        to pixels. Nothing by default: a column nobody wrote this for simply has no row on the table.
        """
        return None


type Collator = Callable[[Sequence[Sample]], Batch]

type Table = pd.DataFrame


class TableSource(ABC):
    """Where annotation rows come from: read once into a table."""

    @abstractmethod
    def read(self) -> Table:
        raise NotImplementedError


class Preprocessor(ABC):
    """The data operations prediction needs, without sources or splits; one object serves training and inference."""

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Resolved input/target facts, including those required to interpret predictions."""

    @property
    def geometries(self) -> Mapping[str, Mapping[str, Geometry]]:
        """What moves with the image under a spatial transform, per ``inputs``/``targets``/``auxiliary_inputs``."""
        return {Role.INPUTS: {}, Role.TARGETS: {}, Role.AUXILIARY: {}}

    @abstractmethod
    def preprocess(self, sample: Sample, transform: SampleTransform | None = None) -> Sample:
        """Load inputs and targets, apply the transform to the loaded values, then encode; never fit here."""

    @abstractmethod
    def collate(self, samples: Sequence[Sample]) -> Batch:
        """Collate already prepared samples, preserving their metadata and explicit count."""

    def fit(self, targets: Mapping[str, Iterable[object]]) -> None:
        """Fit target encoders on raw training cells, keyed by target name; stateless preprocessors ignore this."""
        return None

    def validate(self, targets: Mapping[str, Iterable[object]]) -> None:
        """Check raw cells of another split against the fitted encoders, without refitting."""
        return None

    def describe(self, targets: Mapping[str, Iterable[object]]) -> Mapping[str, Distribution]:
        """What one split's raw target cells hold, per target that can say — the third reading of them.

        The same shape as ``fit`` and ``validate`` take, and for the same reason: the encoders are
        here, and raw cells keyed by target name is what a split hands over. A target whose encoder
        describes nothing is simply absent from the answer.
        """
        return {}

    def warm(self, samples: Iterable[Sample], label: str) -> None:
        """Load raw samples once so a cache can hold them; a preprocessor without a cache does nothing."""
        return None


class DataModule(ABC):
    """Source and split lifecycle; the training adapter owns DataLoader construction."""

    @property
    @abstractmethod
    def preprocessor(self) -> Preprocessor:
        """The prepared encoders and collator; train augmentation belongs to train datasets."""

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Preprocessor facts plus the splits this module prepared."""

    def warm(self, splits: Sequence[str]) -> None:
        """Feed the prepared rows of these splits to the preprocessor's cache; after setup, before workers start."""
        return None

    @abstractmethod
    def setup(self, splits: Sequence[str]) -> None:
        """Prepare the requested splits; neither implicit fitting nor a required train split."""

    def fit_preprocessing(self, train_split: str) -> None:
        """Fit encoders on the named split only, then validate the other prepared splits."""
        return None

    @abstractmethod
    def dataset(self, split: str) -> Dataset[Sample]:
        """Prepared samples of one split: the preprocessor with that split's transform.

        Map-style: the loaders shuffle a training split and hand each device its own share of an
        evaluation one, and both reach a row by index. A stream was named here once and refused by the
        adapter that reads this, which is a contract promising what nothing keeps; it comes back with
        the sampling policy that makes one usable, not before.
        """

    def statistics(self) -> DatasetStatistics:
        """How much of each split there is and what its targets hold, for the report before epoch one.

        Concrete with an empty default: a pipeline that cannot describe its data answers with nothing,
        and whatever asked says so rather than failing.
        """
        return DatasetStatistics()
