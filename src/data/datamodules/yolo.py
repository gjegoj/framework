"""The native YOLO layout behind the ``DataModule`` port."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import torch

from src.core.entities import Batch, DataProfile, Instances, TargetFacts
from src.core.ports import DataModule
from src.core.taxonomy import Modality, Stage

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Dataset

    from src.core.entities import Sample

_STACKED_TENSORS = frozenset({"img", "bboxes", "cls", "batch_idx"})
"""Keys translated into ``Batch`` fields; whatever else ultralytics carries becomes meta."""


class YoloDataModule(DataModule):
    """Per-stage ultralytics datasets built from a YOLO ``data.yaml``.

    A detection dataset does not arrive as an annotation table: it arrives as a
    descriptor beside ``images/`` and ``labels/`` directories, and ultralytics' own
    dataset class carries the box-aware augmentation — mosaic, HSV, perspective — that
    would be expensive and pointless to rebuild. So this delegates to it rather than
    bending the table contour around it.

    It forks nothing else. ``setup`` records the class facts into the ``DataProfile``,
    so a head sizes itself the way it does for any other task, and the run's loader
    knobs keep applying. The one thing it says for itself is batching: detection targets
    are ragged, and the framework's stacking collate cannot serve them.

    ``ultralytics`` is imported inside the methods that need it, so a run that never
    touches detection does not pay for the import.

    Parameters:
        data_yaml (str): Path to the YOLO dataset descriptor — the class list, and one
            image directory per stage. A descriptor with no ``test`` entry simply has
            no test stage, which is ordinary YOLO practice.
        task_name (str): The framework task these classes belong to — the key the
            facts are recorded under, so the head sizes itself from the same profile
            every other head reads. Named ``task_name`` rather than ``task`` because
            ultralytics' own configuration has a ``task`` of its own
            (``detect``/``segment``/``pose``) arriving through ``**hyperparameters``;
            one word for both would let ours swallow theirs, and a segment run could
            not say it was one.
        image_size (int): Square training size (ultralytics ``imgsz``).
        batch_size (int): Batch size the dataset is built for; ultralytics uses
            it while grouping images of similar aspect ratio.
        **hyperparameters (Any): Forwarded verbatim to ultralytics — the
            augmentation knobs (``mosaic``, ``hsv_h``, ``degrees``, ...).
    """

    def __init__(
        self,
        data_yaml: str,
        task_name: str = "detection",
        image_size: int = 640,
        batch_size: int = 16,
        **hyperparameters: Any,
    ) -> None:
        self._data_yaml = data_yaml
        self._task_name = task_name
        self._image_size = image_size
        self._batch_size = batch_size
        self.hyperparameters = hyperparameters
        self._datasets: dict[Stage, Dataset[Sample]] | None = None

    @override
    def setup(self, profile: DataProfile) -> None:
        """Build a dataset per stage and record what the descriptor declares."""
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        # get_cfg is annotated as SimpleNamespace but returns the richer namespace
        # build_yolo_dataset expects — an ultralytics typing mismatch, not ours.
        settings = cast("Any", get_cfg(overrides={"imgsz": self._image_size, **self.hyperparameters}))
        descriptor = check_det_dataset(self._data_yaml)
        names = list(descriptor["names"].values())
        profile.record(self._task_name, TargetFacts(num_classes=len(names), class_names=names))

        def build(path: str, mode: str) -> Any:
            return build_yolo_dataset(settings, img_path=path, batch=self._batch_size, data=descriptor, mode=mode)

        datasets: dict[Stage, Dataset[Sample]] = {
            Stage.TRAIN: build(descriptor["train"], "train"),
            Stage.VAL: build(descriptor["val"], "val"),
        }
        # No `test:` key is ordinary YOLO practice. Serving val in its place is a
        # decision about what a test metric means, so `TrainingData` makes it out loud
        # for every pipeline alike rather than this one making it quietly.
        declared_test = descriptor.get("test")
        if declared_test:
            datasets[Stage.TEST] = build(declared_test, "val")
        self._datasets = datasets

    @override
    def dataset(self, stage: Stage) -> Dataset[Sample]:
        """Return the dataset for ``stage``; ``setup`` must have run first."""
        if self._datasets is None:
            raise RuntimeError("YoloDataModule.setup(profile) must run before requesting datasets.")
        try:
            return self._datasets[stage]
        except KeyError:
            available = ", ".join(self._datasets)
            raise LookupError(f"No dataset for stage '{stage}'. Available stages: {available}.") from None

    @property
    @override
    def collate(self) -> Callable[[list[Sample]], Batch] | None:
        """Ultralytics' own batching, handed on in the framework's currency.

        The translation is the point of an adapter: everything downstream keeps
        receiving a ``Batch``, so the model port does not have to learn a second
        shape. The objects arrive as one ``Instances`` under the task's own name —
        keyed like every other target — concatenated across the batch, because that
        is the only shape a ragged quantity has that a tensor can carry.
        """
        if self._datasets is None:
            raise RuntimeError("YoloDataModule.setup(profile) must run before its batching is known.")
        stack = cast("Any", self._datasets[Stage.TRAIN]).collate_fn

        def collate(samples: list[Sample]) -> Batch:
            from ultralytics.utils.ops import xywhn2xyxy

            stacked = stack(samples)
            raw = stacked["img"]
            # Ultralytics' dataset yields uint8 and its own trainer divides by 255 in
            # `preprocess_batch`, which this contour replaces — so the currency is made
            # right here rather than left for a model to fix.
            pixels = raw.float() / 255 if raw.dtype == torch.uint8 else raw
            height, width = pixels.shape[-2:]
            return Batch(
                inputs={Modality.IMAGE: pixels},
                targets={
                    self._task_name: Instances(
                        # Their normalised cxcywh becomes the framework's xyxy pixels here,
                        # on the side that knows the letterboxed size — so nobody downstream
                        # has to be told it a second time.
                        boxes=xywhn2xyxy(stacked["bboxes"], w=width, h=height),
                        labels=stacked["cls"].reshape(-1).to(torch.int64),
                        sample_index=stacked["batch_idx"].to(torch.int64),
                    )
                },
                meta={key: value for key, value in stacked.items() if key not in _STACKED_TENSORS},
            )

        return collate
