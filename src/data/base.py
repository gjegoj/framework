"""Data contracts: encoders for inputs and targets, the preprocessor that runs them, and the module that owns splits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ClassVar, Protocol, Self, runtime_checkable

import pandas as pd
from torch.utils.data import Dataset, IterableDataset

from src.core import Batch, DatasetInfo, Geometry, InputInfo, Sample, TargetInfo, TensorTree
from src.transforms import SampleTransform


class Encoder(ABC):
    """Raw cell → training value, in two halves around augmentation.

    ``load`` prepares a raw cell for sample transforms (a path becomes pixels); ``encode``
    turns the transformed value into its tensor. ``fit`` learns non-vocabulary state on the
    training split only; ``validate`` checks other splits without refitting. Vocabularies are
    declared, never discovered: an encoder that reads one says so with ``takes_classes`` and
    receives ``classes`` from the task that declares them.
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

    def fit(self, values: Iterable[object]) -> Self:
        return self

    def validate(self, values: Iterable[object]) -> None:
        return None


class InputEncoder(Encoder):
    @property
    @abstractmethod
    def info(self) -> InputInfo:
        """Shape and modality of the encoded value, for heads, export examples and pages."""


class TargetEncoder(Encoder):
    @property
    @abstractmethod
    def info(self) -> TargetInfo:
        """Resolved target facts; available after ``fit`` for encoders that learn a layout."""


@runtime_checkable
class Stateful(Protocol):
    """Optional fitted-state I/O for encoders: learned preprocessing values, never weights or data."""

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


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
        """What moves with the picture under a spatial transform, per ``inputs``/``targets``/``auxiliary_inputs``."""
        return {"inputs": {}, "targets": {}, "auxiliary_inputs": {}}

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
    def dataset(self, split: str) -> Dataset[Sample] | IterableDataset[Sample]:
        """Prepared samples of one split: the preprocessor with that split's transform."""
