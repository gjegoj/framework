"""Lightning's side of the data: prepared splits in, one DataLoader per stage out.

The pipeline is set up before this exists — sources read, encoders fitted, cache warmed — so the only
decisions left here are the ones a loop makes: which split a stage reads, whether its order matters,
and what the run declared about loading. Everything about *what* a sample is stays in ``data``.
"""

from __future__ import annotations

from typing import Any, override

import lightning as L
from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSampler
from torch.utils.data import DataLoader, Dataset, Sampler

from src.core import DatasetInfo, DatasetStatistics, Sample, Stage
from src.data import DataModule, single_threaded_cv2


class TrainingData(L.LightningDataModule):
    """Serves per-stage loaders from an already prepared ``DataModule``.

    ``shuffle`` and ``drop_last`` are conventions of the stage rather than options of the run:
    training shuffles and may drop an incomplete tail, evaluation does neither, because a report is
    about every sample it was given. A stage reads the split of its own name.

    Across several devices the same holds, which takes a sampler of our own: see :meth:`_share_of`.

    Parameters:
        data: The prepared pipeline; ``setup`` has already run.
        **loader_options: Forwarded to every ``DataLoader`` — batch size, workers, pinning.
            ``worker_init_fn`` defaults to :func:`single_threaded_cv2`; a run that declares its own keeps it.
    """

    def __init__(self, data: DataModule, **loader_options: Any) -> None:
        super().__init__()
        self._data = data
        self._drop_last = bool(loader_options.pop("drop_last", False))
        # A default rather than a decree, though from config none can arrive: YAML holds no callables.
        loader_options.setdefault("worker_init_fn", single_threaded_cv2)
        self._options = loader_options

    @property
    def info(self) -> DatasetInfo:
        """What the prepared pipeline settled, for whatever the loop attaches that needs to know.

        A page draws a picture as the file held it, which means undoing the statistics the run applied
        — and those are declared by the input itself, not by the display. This is how a callback
        reaches them: through the object Lightning already hands it, rather than by being told twice.
        """
        return self._data.info

    def statistics(self) -> DatasetStatistics:
        """What the prepared splits hold, for the report a run can print before its first epoch."""
        return self._data.statistics()

    @override
    def train_dataloader(self) -> DataLoader[Sample]:
        return self._loader(Stage.TRAIN, shuffle=True, drop_last=self._drop_last)

    @override
    def val_dataloader(self) -> DataLoader[Sample]:
        return self._loader(Stage.VAL, shuffle=False, drop_last=False)

    @override
    def test_dataloader(self) -> DataLoader[Sample]:
        return self._loader(Stage.TEST, shuffle=False, drop_last=False)

    def _loader(self, stage: Stage, *, shuffle: bool, drop_last: bool) -> DataLoader[Sample]:
        dataset = self._data.dataset(stage)
        return DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            # Shuffling and a sampler are alternatives to a DataLoader, and only training shuffles:
            # how training spreads over devices stays Lightning's, for the reason `_share_of` gives.
            sampler=None if shuffle else self._share_of(dataset),
            # The pipeline that prepared the samples is what joins them: a modality whose samples are
            # ragged collates its own way, and this adapter never learns which one it is serving.
            collate_fn=self._data.preprocessor.collate,
            **self._options,
        )

    def _share_of(self, dataset: Dataset[Sample]) -> Sampler[int] | None:
        """This device's part of an evaluation split, with no row landing on two of them.

        Lightning inserts a distributed sampler of its own, and that one pads the last batch by
        wrapping around to the start — right for training, where every device owes the same number of
        gradient steps and a repeated sample is one shuffled draw among many, and wrong for a report,
        where those rows are scored twice. Measured on lightning 2.6.5: a loader that already carries
        a ``DistributedSampler`` is left alone, and this is one, which is how the swap is made.

        The cost is that devices finish an evaluation a batch apart. That is safe while nothing in the
        loop is collective per batch — metrics gather when they compute, and this framework logs
        nothing with ``sync_dist`` on a step — and it is why Lightning uses this same sampler to predict.
        """
        trainer = self.trainer
        if trainer is None or trainer.world_size == 1:
            return None
        return UnrepeatedDistributedSampler(
            dataset, num_replicas=trainer.world_size, rank=trainer.global_rank, shuffle=False
        )
