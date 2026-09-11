"""Stage datasets in, DataLoaders out: the one adapter between a prepared pipeline and Lightning.

What is under test is the handful of conventions a loop relies on — which split a stage reads, who
makes the batch, and which options are the stage's own rather than the run's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from src.core import Batch, DatasetInfo, Sample
from src.data import DataModule, Preprocessor, single_threaded_cv2
from src.training import TrainingData

SPLITS = {"train": 6, "val": 4, "test": 2}


class Rows(Dataset[Sample]):
    """As many samples as the split has, each carrying the number it sits at."""

    def __init__(self, split: str, size: int) -> None:
        self.split, self.size = split, size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Sample:
        return Sample(inputs={"row": torch.tensor([index])}, metadata={"split": self.split})


class Marked(Preprocessor):
    """A preprocessor whose collate is recognisable, so a batch can say who made it."""

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(inputs={}, targets={})

    def preprocess(self, sample: Sample, transform: object = None) -> Sample:
        return sample

    def collate(self, samples: Sequence[Sample]) -> Batch:
        return Batch(
            inputs={"row": torch.stack([torch.as_tensor(one.inputs["row"]) for one in samples])},
            count=len(samples),
            metadata={"collated_by": "preprocessor", "split": samples[0].metadata["split"]},
        )


class Prepared(DataModule):
    """A module whose splits are already prepared: this is a test about loaders, not about sources."""

    def __init__(self) -> None:
        self._preprocessor = Marked()

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(inputs={}, targets={}, splits=tuple(SPLITS))

    def setup(self, splits: Sequence[str]) -> None:
        return None

    def dataset(self, split: str) -> Dataset[Sample]:
        if split not in SPLITS:
            raise LookupError(f"No split named {split!r}.")
        return Rows(split, SPLITS[split])


@pytest.fixture
def data() -> TrainingData:
    return TrainingData(Prepared(), batch_size=2)


def first(batches: Iterable[Batch]) -> Batch:
    return next(iter(batches))


class TestStages:
    @pytest.mark.parametrize("stage", list(SPLITS))
    def test_every_stage_reads_the_split_of_its_own_name(self, data: TrainingData, stage: str) -> None:
        loaders: Mapping[str, object] = {
            "train": data.train_dataloader,
            "val": data.val_dataloader,
            "test": data.test_dataloader,
        }

        batch = first(loaders[stage]())  # type: ignore[operator]

        assert batch.metadata["split"] == stage

    def test_a_batch_is_made_by_the_preprocessor_that_prepared_the_samples(self, data: TrainingData) -> None:
        """Collation belongs to the pipeline: a ragged modality collates its own way, and this adapter never knows."""
        assert first(data.train_dataloader()).metadata["collated_by"] == "preprocessor"

    def test_training_shuffles_and_evaluation_keeps_the_order_it_was_given(self, data: TrainingData) -> None:
        assert isinstance(data.train_dataloader().sampler, RandomSampler)
        assert isinstance(data.val_dataloader().sampler, SequentialSampler)
        assert isinstance(data.test_dataloader().sampler, SequentialSampler)


class TestOptions:
    def test_only_training_may_drop_an_incomplete_batch(self) -> None:
        """Evaluation reports on every sample it was given; a dropped tail would be a silently smaller test."""
        data = TrainingData(Prepared(), batch_size=4, drop_last=True)

        assert data.train_dataloader().drop_last is True
        assert data.val_dataloader().drop_last is False

    def test_whatever_else_a_run_declares_reaches_every_loader(self, data: TrainingData) -> None:
        declared = TrainingData(Prepared(), batch_size=2, num_workers=2, pin_memory=True)

        assert declared.val_dataloader().num_workers == 2
        assert declared.val_dataloader().pin_memory is True
        assert data.val_dataloader().num_workers == 0

    def test_workers_decode_on_one_thread_each_unless_the_run_says_otherwise(self, data: TrainingData) -> None:
        """Workers are the parallelism; cv2's own pool would run them against each other."""

        def mine(worker: int) -> None:
            return None

        assert data.train_dataloader().worker_init_fn is single_threaded_cv2
        assert TrainingData(Prepared(), worker_init_fn=mine).train_dataloader().worker_init_fn is mine
