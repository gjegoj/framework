"""The native YOLO layout behind the ``DataModule`` port."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import torch

from src.core.entities import Batch, DataProfile, Instances, TargetFacts
from src.core.ports import DataModule, require_stage
from src.core.taxonomy import Modality, Stage
from src.data.registry import vendor_data_module_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import Dataset

    from src.core.entities import Sample

_STACKED_TENSORS = frozenset({"img", "bboxes", "cls", "batch_idx"})
"""Keys translated into ``Batch`` fields; whatever else ultralytics carries becomes meta."""


@vendor_data_module_registry.register("yolo")
class YoloDataModule(DataModule):
    """Per-stage ultralytics datasets built from a YOLO ``data.yaml``.

    Registered under the same key ``YoloModel`` takes in ``vendor_model_registry``, so a run
    names the family once. Delegates to ultralytics' own dataset for its box-aware
    augmentation (mosaic, HSV, perspective); ``setup`` records the class facts into the
    ``DataProfile`` so a head sizes itself as for any other task. ``ultralytics`` is imported
    inside the methods that need it.

    Parameters:
        data_yaml (str): Path to the YOLO dataset descriptor — the class list, and one image
            directory per stage. No ``test`` entry means no test stage.
        task_name (str): The framework task these classes belong to. Not ``task``: ultralytics'
            own ``task`` (``detect``/``segment``/``pose``) arrives through ``**hyperparameters``.
        image_size (int): Square training size (ultralytics ``imgsz``).
        batch_size (int): Batch size the dataset is built for.
        **hyperparameters (Any): Forwarded verbatim to ultralytics — ``mosaic``, ``hsv_h``, ...
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

        # The descriptor's stage keys are the framework's own names; ``mode`` is
        # ultralytics' vocabulary and stays spelled as it reads it.
        datasets: dict[Stage, Dataset[Sample]] = {
            Stage.TRAIN: build(descriptor[Stage.TRAIN], mode="train"),
            Stage.VAL: build(descriptor[Stage.VAL], mode="val"),
        }
        # No `test:` key is ordinary YOLO practice. Serving val in its place is a
        # decision about what a test metric means, so `TrainingData` makes it out loud
        # for every pipeline alike rather than this one making it quietly.
        declared_test = descriptor.get(Stage.TEST)
        if declared_test:
            datasets[Stage.TEST] = build(declared_test, mode="val")
        self._datasets = datasets

    @override
    def dataset(self, stage: Stage) -> Dataset[Sample]:
        """Return the dataset for ``stage``; ``setup`` must have run first."""
        return require_stage(self._datasets, stage, type(self).__name__)

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
