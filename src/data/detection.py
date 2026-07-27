"""Detection data module over ultralytics ``YOLODataset`` (native YOLO dataset format).

The CSV/bindings ``DataModule`` is deliberately not bent around YOLO's directory
format: detection datasets arrive as ``data.yaml`` + images/labels dirs, and
ultralytics' own dataset class carries the box-aware augmentation pipeline
(mosaic/HSV/perspective) that is expensive to rebuild. Migrating detection onto the
bindings contour is Option-A future work (see the detection design spec).
"""

from __future__ import annotations

from typing import Any, cast

import lightning as L
from torch.utils.data import DataLoader, Dataset


class DetectionDataModule(L.LightningDataModule):
    """Builds train/val ``YOLODataset``s from a YOLO ``data.yaml`` descriptor.

    The test stage reuses the val split when the descriptor has no ``test`` entry
    (standard YOLO practice). ``num_classes``/``class_names`` are read from the
    descriptor at ``setup()``.

    Parameters:
        data_yaml (str): Path to the YOLO dataset descriptor (``data.yaml``).
        image_size (int): Square training image size (ultralytics ``imgsz``).
        batch_size (int): Batch size for every stage.
        hyperparameters (dict[str, Any] | None): ultralytics hyp overrides forwarded
            verbatim (augmentation knobs like ``mosaic``/``hsv_h``/``degrees``).
        num_workers (int): DataLoader worker processes.
    """

    def __init__(
        self,
        data_yaml: str,
        image_size: int,
        batch_size: int,
        hyperparameters: dict[str, Any] | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self._data_yaml = data_yaml
        self._image_size = image_size
        self._batch_size = batch_size
        self._hyperparameters = dict(hyperparameters or {})
        self._num_workers = num_workers
        self._train_dataset: Dataset[Any] | None = None
        self._val_dataset: Dataset[Any] | None = None
        self._test_dataset: Dataset[Any] | None = None
        self._class_names: list[str] = []

    @property
    def num_classes(self) -> int:
        """Detection class count, read from ``data.yaml`` at ``setup()``."""
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        """Ordered class names, read from ``data.yaml`` at ``setup()``."""
        return list(self._class_names)

    def setup(self, stage: str | None = None) -> None:
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        # get_cfg is annotated as SimpleNamespace but returns the richer IterableSimpleNamespace
        # that build_yolo_dataset expects — an ultralytics typing mismatch, not ours.
        configuration = cast("Any", get_cfg(overrides={"imgsz": self._image_size, **self._hyperparameters}))
        descriptor = check_det_dataset(self._data_yaml)
        self._class_names = list(descriptor["names"].values())
        self._train_dataset = build_yolo_dataset(
            configuration, img_path=descriptor["train"], batch=self._batch_size, data=descriptor, mode="train"
        )
        self._val_dataset = build_yolo_dataset(
            configuration, img_path=descriptor["val"], batch=self._batch_size, data=descriptor, mode="val"
        )
        test_path = descriptor.get("test")
        if test_path:
            self._test_dataset = build_yolo_dataset(
                configuration, img_path=test_path, batch=self._batch_size, data=descriptor, mode="val"
            )
        else:
            self._test_dataset = self._val_dataset  # standard YOLO practice: no test split -> evaluate on val

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self._require(self._train_dataset, "train"), shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self._require(self._val_dataset, "val"), shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self._require(self._test_dataset, "test"), shuffle=False)

    def _dataloader(self, dataset: Dataset[Any], *, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=getattr(dataset, "collate_fn"),  # noqa: B009 — the YOLODataset attr is invisible to typing
        )

    @staticmethod
    def _require(dataset: Dataset[Any] | None, stage: str) -> Dataset[Any]:
        if dataset is None:
            raise RuntimeError(f"DetectionDataModule.setup() must run before requesting the {stage} dataloader.")
        return dataset
