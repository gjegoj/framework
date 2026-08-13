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

    OpenCV carries a process-wide thread pool sized to the machine, and loader
    workers are processes — left alone, eight workers on eight cores run
    sixty-four decoding threads against eight ancestors of them. Here the workers
    *are* the parallelism, so each one decodes single-threaded; an epoch on an
    8-core box measured 1.47x faster for it. The setting also reaches every other
    cv2 call in the worker, albumentations' included.

    A ``worker_init_fn`` is the one placement that survives every start method:
    the pool is per-process and rebuilt on ``spawn``, so setting it in the parent
    works only while ``fork`` lets children inherit it, and OpenCV reads no
    environment variable that could travel instead. Workers-only is also the
    right scope — a ``num_workers: 0`` run keeps cv2's own parallelism, which is
    all the parallelism it has.
    """
    cv2.setNumThreads(0)


class TrainingData(L.LightningDataModule):
    """Serves per-stage DataLoaders from an already-set-up ``DataModule``.

    Loader options forward verbatim to ``torch.utils.data.DataLoader``, so any
    torch knob is reachable without this class declaring it. Two arguments are
    the adapter's own and are not accepted among them: ``shuffle`` and
    ``drop_last`` are stage conventions — training shuffles and may drop its
    last incomplete batch, evaluation does neither, because a dropped
    evaluation batch means metrics computed on part of the split.

    Setup ordering lives with assembly: profile facts must exist before the
    model is built, so ``DataModule.setup`` runs there, eagerly. This adapter
    only turns stage datasets into loaders.

    Parameters:
        data (DataModule): Source of per-stage datasets; ``setup`` has run.
        collate (Callable | None): Turns samples into a ``Batch``; ``None`` takes
            the framework's own. A pipeline with ragged targets — detection
            boxes, one image carrying three and the next eleven — reports its
            own through ``DataModule.collate``, and assembly passes it here.
        **loader_options (Any): Forwarded to every ``DataLoader``. One default is
            filled in: ``worker_init_fn`` is :func:`single_threaded_cv2` above,
            unless the caller passes their own — see its docstring for the why.
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

        A pipeline may honestly have none: a YOLO descriptor often ships without a
        test split, and per-stage sources need not declare all three. Falling back
        lets such a run finish and report something, instead of dying after the fit
        with the weights already trained.

        What it must not do is stay quiet. Every ``test/*`` scalar would then be
        computed on the rows the checkpoint was selected on and published under the
        one name that is supposed to mean held-out data — an optimistic number
        wearing an honest label. So the substitution is named once, here.

        Resolved at the one place that asks for a test loader rather than inside
        each pipeline, so every datamodule falls back the same way and says the same
        sentence.
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
