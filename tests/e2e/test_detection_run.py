"""End to end: a detection run on the table pipeline — the composed family, from canon rows to mAP.

Red on purpose until the detection criterion lands (roadmap stages 3–4): the model builds,
and the kind refuses to train it because no criterion reads its raw tensor yet. ``xfail(strict=True)`` is what makes the day the
criterion lands loud — the test passes, the marker fails, and somebody removes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.build import build, run
from tests.support.detection import annotation_tree, detection_config

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("ultralytics", reason="the detection family parses ultralytics' graphs")

ULTRALYTICS = {"name": "ultralytics", "model_name": "yolov8n.yaml"}


@pytest.mark.xfail(
    strict=True, raises=ValueError, reason="roadmap stages 3–4: the composed family has no criterion yet"
)
def test_a_detection_run_on_the_table_pipeline_trains_and_reports_map(tmp_path: Path) -> None:
    """One epoch on four images: the loss parts log by name and the metric family appears."""
    annotation_tree(tmp_path)
    config = detection_config(
        tmp_path,
        model=ULTRALYTICS,
        image_size=[64, 64],
        loader={"batch_size": 2},
        trainer={"max_epochs": 1, "accelerator": "cpu", "logger": False, "default_root_dir": str(tmp_path)},
        run={"directory": str(tmp_path / "run"), "test": False},
    )
    experiment = build(config)

    run(experiment, config)

    logged = set(experiment.trainer.callback_metrics)
    assert {"train/boxes/box", "train/boxes/cls", "train/boxes/dfl"} <= logged
    assert {"val/boxes/map/map", "val/boxes/map/map_50", "val/boxes/map/map_75"} <= logged
