"""Lightning's side of the data: prepared splits in, one DataLoader per stage out.

The pipeline is set up before this exists — sources read, encoders fitted, cache warmed — so the only
decisions left here are the ones a loop makes: which split a stage reads, whether its order matters,
and what the run declared about loading. Everything about *what* a sample is stays in ``data``.
"""

from __future__ import annotations

from typing import Any, override

import lightning as L
from torch.utils.data import DataLoader

from src.core import Sample, Stage
from src.data import DataModule, single_threaded_cv2


class TrainingData(L.LightningDataModule):
    """Serves per-stage loaders from an already prepared ``DataModule``.

    ``shuffle`` and ``drop_last`` are conventions of the stage rather than options of the run:
    training shuffles and may drop an incomplete tail, evaluation does neither, because a report is
    about every sample it was given. A stage reads the split of its own name.

    On one device that is exactly true. Across several, Lightning inserts a distributed sampler that
    pads the last batch by wrapping around, so a few rows are scored twice; run the final evaluation
    on one device when a report has to be exact.

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
        return DataLoader(
            self._data.dataset(stage),
            shuffle=shuffle,
            drop_last=drop_last,
            # The pipeline that prepared the samples is what joins them: a modality whose samples are
            # ragged collates its own way, and this adapter never learns which one it is serving.
            collate_fn=self._data.preprocessor.collate,
            **self._options,
        )
