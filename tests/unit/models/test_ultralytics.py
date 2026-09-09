"""The ultralytics graph behind our ports: a pyramid backbone and a Detect head, weights grafted loudly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from src.core import Stream
from src.models.backbones.ultralytics import UltralyticsBackbone
from src.models.heads import DetectHead
from src.models.registry import backbone_registry

THREE = ("p3", "p4", "p5")


def test_the_family_is_reachable_from_config_by_name() -> None:
    assert isinstance(backbone_registry.create("ultralytics", model_name="yolov8n.yaml"), UltralyticsBackbone)


def test_the_pyramid_is_named_by_stride_with_the_measured_widths() -> None:
    """Measured on ultralytics 8.4.115: yolov8n feeds Detect from 64/128/256 channels at strides 8/16/32."""
    backbone = UltralyticsBackbone("yolov8n.yaml")

    assert backbone.pyramid() == THREE
    assert backbone.feature_dims() == {"p3": 64, "p4": 128, "p5": 256, Stream.FEATURES: 256}
    assert backbone.strides == (8, 16, 32)
    assert backbone.architecture == "yolov8n"


def test_a_four_level_yaml_declares_four_levels_without_a_word_of_config() -> None:
    """Measured: ``yolov8n-p2.yaml`` reads strides 4/8/16/32 — the count and names come from the graph."""
    backbone = UltralyticsBackbone("yolov8n-p2.yaml")

    assert backbone.pyramid() == ("p2", "p3", "p4", "p5")
    assert backbone.strides == (4, 8, 16, 32)


def test_a_forward_yields_the_levels_and_the_pooled_coarsest_one() -> None:
    features = UltralyticsBackbone("yolov8n.yaml")({"image": torch.zeros(2, 3, 64, 64)})

    assert tuple(features["p3"].shape) == (2, 64, 8, 8)
    assert tuple(features["p4"].shape) == (2, 128, 4, 4)
    assert tuple(features["p5"].shape) == (2, 256, 2, 2)
    assert tuple(features[Stream.FEATURES].shape) == (2, 256)


def test_the_native_head_is_a_detect_head_sized_for_the_task() -> None:
    backbone = UltralyticsBackbone("yolov8n.yaml")

    head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 3)

    assert isinstance(head, DetectHead)
    assert (head.num_classes, head.reg_max, head.strides, head.streams) == (3, 16, (8, 16, 32), THREE)
    assert backbone.native_head((Stream.FEATURES,), 256, 3) is None


@pytest.mark.parametrize("training", [True, False])
def test_the_head_returns_one_raw_tensor_in_every_mode(training: bool) -> None:
    """``[B, 4·reg_max + nc, A]`` — decoding is a separate step, not a mode; A = Σ (H/s)(W/s)."""
    backbone = UltralyticsBackbone("yolov8n.yaml")
    head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 3)
    assert head is not None
    head.train(training)
    features = backbone({"image": torch.zeros(2, 3, 64, 64)})

    logits = head({name: features[name] for name in THREE})

    assert tuple(logits.shape) == (2, 4 * 16 + 3, 8 * 8 + 4 * 4 + 2 * 2)


def test_a_four_level_head_reads_all_four() -> None:
    backbone = UltralyticsBackbone("yolov8n-p2.yaml")
    widths = tuple(backbone.feature_dims()[name] for name in backbone.pyramid())
    head = backbone.native_head(backbone.pyramid(), widths, 3)
    assert head is not None
    features = backbone({"image": torch.zeros(1, 3, 64, 64)})

    logits = head({name: features[name] for name in backbone.pyramid()})

    assert tuple(logits.shape) == (1, 4 * 16 + 3, 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2)


def test_the_head_refuses_a_single_stream_by_name() -> None:
    backbone = UltralyticsBackbone("yolov8n.yaml")
    head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 3)
    assert head is not None

    with pytest.raises(TypeError, match="DetectHead reads the pyramid p3, p4, p5"):
        head(torch.zeros(1, 256, 2, 2))


def test_the_template_head_never_enters_the_module_tree() -> None:
    """Its weights never run in forward, so they must not reach checkpoints or exports."""
    keys = list(UltralyticsBackbone("yolov8n.yaml").state_dict())

    assert keys[0] == "model.0.conv.weight"
    assert {key.split(".")[0] for key in keys} == {"model"}  # nothing registered beside the body
    assert not any(key.startswith("model.22.") for key in keys)


@pytest.mark.parametrize(
    ("model_name", "expected"), [("yolov10n.yaml", "v10Detect"), ("rtdetr-l.yaml", "RTDETRDecoder")]
)
def test_a_head_outside_the_detect_line_is_refused_naming_it(model_name: str, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        UltralyticsBackbone(model_name)


def test_weights_never_come_with_the_model_name() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        UltralyticsBackbone("yolov8n.pt")


def saved_graph(path: Path, model_name: str = "yolov8n.yaml", classes: int = 3) -> Path:
    """What ultralytics writes: a pickled ``DetectionModel`` under ``model``."""
    from ultralytics.nn.tasks import DetectionModel

    graph: Any = DetectionModel(model_name, nc=classes, verbose=False)
    with torch.no_grad():
        graph.model[0].conv.weight.fill_(0.5)  # a body tensor to recognise
        graph.model[-1].cv2[0][-1].bias.fill_(0.25)  # box branch: transplants whatever nc
        graph.model[-1].cv3[0][-1].bias.fill_(-0.75)  # class branch: transplants only on equal nc
    torch.save({"model": graph, "ema": None, "train_args": {}}, path)
    return path


def test_a_checkpoint_loads_the_body_strictly_and_stashes_the_head(tmp_path: Path) -> None:
    backbone = UltralyticsBackbone("yolov8n.yaml", checkpoint_path=saved_graph(tmp_path / "w.pt"))

    first: Any = backbone.model[0]
    assert torch.all(first.conv.weight == 0.5)


def test_the_box_branch_transplants_whatever_the_class_count(tmp_path: Path) -> None:
    backbone = UltralyticsBackbone("yolov8n.yaml", checkpoint_path=saved_graph(tmp_path / "w.pt", classes=3))

    head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 5)
    assert isinstance(head, DetectHead)

    detect: Any = head.detect
    assert torch.all(detect.cv2[0][-1].bias == 0.25)
    assert not torch.any(detect.cv3[0][-1].bias == -0.75)  # 5 classes: a fresh class branch


def test_the_class_branch_transplants_on_the_same_class_count_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backbone = UltralyticsBackbone("yolov8n.yaml", checkpoint_path=saved_graph(tmp_path / "w.pt", classes=3))

    with caplog.at_level(logging.INFO, logger="src.models.backbones.ultralytics"):
        head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 3)
    assert isinstance(head, DetectHead)

    detect: Any = head.detect
    assert torch.all(detect.cv3[0][-1].bias == -0.75)
    assert "Transplanted 85 of 85 head tensors" in caplog.text


def test_a_checkpoint_of_another_architecture_is_refused_naming_both(tmp_path: Path) -> None:
    """The message names the file and the architecture it did not fit."""
    with pytest.raises(ValueError, match="s.pt.*yolov8n"):
        UltralyticsBackbone("yolov8n.yaml", checkpoint_path=saved_graph(tmp_path / "s.pt", model_name="yolov8s.yaml"))


def test_an_end_to_end_detect_head_is_refused_naming_the_line() -> None:
    """yolo26's head is a ``Detect`` too, with ``end2end=True`` and ``reg_max=1``; rebuilt at the
    defaults it was silently another head — measured: 7 channels expected, 67 produced."""
    with pytest.raises(ValueError, match="end-to-end"):
        UltralyticsBackbone("yolo26n.yaml")


def test_the_rebuilt_head_keeps_the_templates_reg_max(tmp_path: Path) -> None:
    """``reg_max`` is a fact of the yaml, read off the template; the rebuilt head takes it from
    there rather than from ultralytics' default."""
    import ultralytics

    shipped = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    yaml = tmp_path / "yolov8n-dfl8.yaml"
    yaml.write_text("reg_max: 8\n" + shipped.read_text(encoding="utf-8"), encoding="utf-8")
    backbone = UltralyticsBackbone(str(yaml))

    head = backbone.native_head(backbone.pyramid(), (64, 128, 256), 3)
    assert isinstance(head, DetectHead)

    raw = head(backbone({"image": torch.zeros(1, 3, 64, 64)}).streams)
    assert head.reg_max == 8
    assert raw.shape[1] == 4 * 8 + 3


def test_building_the_graph_leaves_the_yamls_class_count_alone(capfd: pytest.CaptureFixture[str]) -> None:
    """The head is dropped from the forward path, so ``nc`` is nobody's business here —
    overriding it only made ultralytics print a line at every build (measured)."""
    UltralyticsBackbone("yolov8n.yaml")

    captured = capfd.readouterr()
    assert "Overriding" not in captured.out + captured.err
