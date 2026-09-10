"""Source and split lifecycle; the training adapter owns DataLoader construction."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch.utils.data import Dataset, IterableDataset

from src.core import DatasetInfo, Sample
from src.data.preprocessing import Preprocessor


class DataModule(ABC):
    @property
    @abstractmethod
    def preprocessor(self) -> Preprocessor:
        """Prepared inference transforms/encoders; train augmentations belong to train datasets."""

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Preprocessor input/target facts plus the available dataset splits."""

    def prepare_data(self) -> None:
        """Optional source download/cache warmup; never fit preprocessing."""

    @abstractmethod
    def setup(self, splits: Sequence[str]) -> None:
        """Prepare requested splits; neither implicit fitting nor a required train split."""

    def fit_preprocessing(self, train_split: str) -> None:
        """Override when preprocessing must be fitted; use only the explicitly selected train split."""

    @abstractmethod
    def dataset(self, split: str) -> Dataset[Sample] | IterableDataset[Sample]:
        """Return prepared samples; apply train geometry between load and encode, never after encode.

        Use the same resolved encoders as preprocessor, with split-specific sample
        transforms. Collation must not apply preprocessing a second time.
        """
