"""``checkpoint_path`` on the smp backbone: arrived weights load, the head transplants."""

from __future__ import annotations

from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch
from torch import nn

from src.models import ExpandedHead, SmpBackbone


def build_backbone(**kwargs: object) -> SmpBackbone:
    return SmpBackbone(arch="unet", encoder_name="resnet18", **kwargs)  # type: ignore[arg-type]


def full_model_file(tmp_path: Path, classes: int = 3) -> tuple[Path, nn.Module]:
    trained = smp.create_model(arch="unet", encoder_name="resnet18", encoder_weights=None, classes=classes)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": trained.state_dict()}, path)
    return path, trained


def test_encoder_and_decoder_tensors_arrive_from_the_checkpoint(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path)

    backbone = build_backbone(checkpoint_path=path)

    assert torch.equal(backbone._encoder.state_dict()["conv1.weight"], trained.state_dict()["encoder.conv1.weight"])
    decoder_key = next(iter(backbone._decoder.state_dict()))
    assert torch.equal(backbone._decoder.state_dict()[decoder_key], trained.state_dict()[f"decoder.{decoder_key}"])


def test_the_segmentation_head_is_stashed_not_loaded(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path)

    backbone = build_backbone(checkpoint_path=path)

    assert backbone._carried_head is not None
    assert torch.equal(
        backbone._carried_head["segmentation_head.0.weight"], trained.state_dict()["segmentation_head.0.weight"]
    )


def test_equal_class_counts_transplant_the_whole_head(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path, classes=3)
    backbone = build_backbone(checkpoint_path=path)

    head = backbone.native_head("decoder", backbone.feature_dim("decoder"), 3)

    assert head is not None
    assert not isinstance(head, ExpandedHead)
    assert torch.equal(head.state_dict()["0.weight"], trained.state_dict()["segmentation_head.0.weight"])


def test_growing_the_class_space_transplants_base_and_leaves_novel_fresh(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path, classes=3)
    backbone = build_backbone(checkpoint_path=path)

    head = backbone.native_head("decoder", backbone.feature_dim("decoder"), 4)

    assert isinstance(head, ExpandedHead)
    assert torch.equal(head.base.state_dict()["0.weight"], trained.state_dict()["segmentation_head.0.weight"])
    novel_weight = head.novel.state_dict()["0.weight"]
    assert novel_weight.shape[0] == 1

    logits = head(torch.randn(2, backbone.feature_dim("decoder"), 8, 8))
    assert logits.shape[:2] == (2, 4)


def test_narrowing_the_class_space_is_refused(tmp_path: Path) -> None:
    path, _ = full_model_file(tmp_path, classes=3)
    backbone = build_backbone(checkpoint_path=path)

    with pytest.raises(ValueError, match="mapping"):
        backbone.native_head("decoder", backbone.feature_dim("decoder"), 2)


def test_a_foreign_decoder_width_is_refused(tmp_path: Path) -> None:
    trained = smp.create_model(arch="unet", encoder_name="resnet18", encoder_weights=None, classes=3)
    state = trained.state_dict()
    state["segmentation_head.0.weight"] = torch.zeros(3, 99, 3, 3)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": state}, path)
    backbone = build_backbone(checkpoint_path=path)

    with pytest.raises(ValueError, match="feature"):
        backbone.native_head("decoder", backbone.feature_dim("decoder"), 3)


def test_without_a_checkpoint_the_native_head_stays_fresh() -> None:
    backbone = build_backbone(pretrained=False)

    head = backbone.native_head("decoder", backbone.feature_dim("decoder"), 3)

    assert head is not None
    assert not isinstance(head, ExpandedHead)
