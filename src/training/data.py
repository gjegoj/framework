"""Lightning-facing data adapter: stage datasets in, DataLoaders out."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

import cv2
import lightning as L
from torch.utils.data import DataLoader

from src.core.taxonomy import Stage
from src.data import collate_samples

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.entities import Batch, Sample
    from src.core.ports import DataModule

log = logging.getLogger(__name__)


def single_threaded_cv2(_worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: one cv2 thread per worker, measured 1.5x faster.

    OpenCV's process-wide thread pool is sized to the machine, and loader workers are
    processes, so eight workers run sixty-four decoding threads; here the workers *are* the
    parallelism. A ``worker_init_fn`` survives every start method, and a ``num_workers: 0``
    run keeps cv2's own parallelism.
    """
    cv2.setNumThreads(0)


class TrainingData(L.LightningDataModule):
    """Serves per-stage DataLoaders from an already-set-up ``DataModule``.

    ``shuffle`` and ``drop_last`` are stage conventions and not accepted among the options:
    training shuffles and may drop its last batch, evaluation does neither.
    ``DataModule.setup`` runs in assembly, eagerly; this only turns stage datasets into loaders.

    Parameters:
        data (DataModule): Source of per-stage datasets; ``setup`` has run.
        collate (Callable | None): Turns samples into a ``Batch``; ``None`` takes the
            framework's own. A pipeline with ragged targets reports its own through
            ``DataModule.collate``.
        **loader_options (Any): Forwarded to every ``DataLoader``; ``worker_init_fn``
            defaults to :func:`single_threaded_cv2`.
    """

    def __init__(
        self,
        data: DataModule,
        collate: Callable[[list[Sample]], Batch] | None = None,
        **loader_options: Any,
    ) -> None:
        super().__init__()
        self._data = data
        self._collate = collate if collate is not None else collate_samples
        self._drop_last = bool(loader_options.pop("drop_last", False))
        # A default, not a decree — a caller's own worker_init_fn wins. From config
        # none can arrive (YAML holds no callables), so every run gets this one.
        loader_options.setdefault("worker_init_fn", single_threaded_cv2)
        self._options = loader_options

    @property
    def source(self) -> DataModule:
        """The pipeline this adapter serves loaders from.

        Public because consumers legitimately need the port rather than the
        adapter: the dataset report asks it for ``statistics()``, and reaching
        through a private attribute to get there would make an accessor out of an
        implementation detail.
        """
        return self._data

    @override
    def train_dataloader(self) -> DataLoader[Sample]:
        return self._loader(Stage.TRAIN, shuffle=True, drop_last=self._drop_last)

    @override
    def val_dataloader(self) -> DataLoader[Sample]:
        return self._loader(Stage.VAL, shuffle=False, drop_last=False)

    @override
    def test_dataloader(self) -> DataLoader[Sample]:
        return self._loader(self._tested_stage(), shuffle=False, drop_last=False)

    def _tested_stage(self) -> Stage:
        """Test, or validation when the run declared no test data — and then it says so.

        A pipeline may honestly have no test split, and falling back lets the run finish. What
        it must not do is stay quiet: every ``test/*`` scalar would be computed on the rows the
        checkpoint was selected on, under the one name that means held-out data. Resolved here
        so every datamodule falls back the same way.
        """
        try:
            self._data.dataset(Stage.TEST)
        except LookupError:
            log.warning(
                "This run declares no test data, so the test stage runs on the validation set — "
                "the same rows the checkpoint was selected on. Every test/* metric is optimistic "
                "for that reason; declare test data to get an honest one."
            )
            return Stage.VAL
        return Stage.TEST

    def _loader(self, stage: Stage, *, shuffle: bool, drop_last: bool) -> DataLoader[Sample]:
        return DataLoader(
            self._data.dataset(stage),
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self._collate,
            **self._options,
        )
