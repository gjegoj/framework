"""The data operations needed for prediction, without sources or dataset splits."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from src.core import Batch, DatasetInfo, Sample


@runtime_checkable
class Stateful(Protocol):
    """Optional fitted-state I/O for encoders and preprocessing adapters.

    Contains learned preprocessing values, not model weights, datasets or caches.
    Artifacts store the construction config separately. A preprocessing adapter
    aggregates its children's state; restoration never fits on new data.
    """

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class Preprocessor(ABC):
    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Resolved input/target facts, including those required to interpret predictions."""

    @abstractmethod
    def preprocess(self, sample: Sample) -> Sample:
        """Load inputs/targets, transform jointly, then encode targets; no fitting here."""

    @abstractmethod
    def collate(self, samples: Sequence[Sample]) -> Batch:
        """Collate already prepared samples, preserving their metadata and explicit count."""
