"""DetectionDataModule: parses data.yaml, yields ultralytics-format batches."""

from __future__ import annotations

from pathlib import Path

from src.data.detection import DetectionDataModule
from tests.support.builders import make_yolo_dataset


def _datamodule(tmp_path: Path) -> DetectionDataModule:
    data_yaml = make_yolo_dataset(tmp_path)
    datamodule = DetectionDataModule(data_yaml=str(data_yaml), image_size=64, batch_size=4)
    datamodule.setup()
    return datamodule


class TestDetectionDataModule:
    def test_setup_reads_classes(self, tmp_path: Path) -> None:
        datamodule = _datamodule(tmp_path)
        assert datamodule.num_classes == 2
        assert datamodule.class_names == ["a", "b"]

    def test_train_loader_yields_ultralytics_batches(self, tmp_path: Path) -> None:
        batch = next(iter(_datamodule(tmp_path).train_dataloader()))
        assert set(batch) >= {"img", "cls", "bboxes", "batch_idx"}
        assert batch["img"].ndim == 4  # [B, C, H, W]

    def test_val_reused_for_test_when_absent(self, tmp_path: Path) -> None:
        datamodule = _datamodule(tmp_path)
        assert datamodule.test_dataloader().dataset is datamodule.val_dataloader().dataset
