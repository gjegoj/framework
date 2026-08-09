"""``checkpoint_path`` on the timm backbone: arrived weights load with timm's own knobs."""

from __future__ import annotations

from pathlib import Path

import pytest
import timm
import torch
from torch import nn

from src.models import ExpandedHead, TimmBackbone

MODEL = "resnet18"


def full_model_file(tmp_path: Path, num_classes: int = 3, key: str = "state_dict") -> tuple[Path, nn.Module]:
    trained = timm.create_model(MODEL, pretrained=False, num_classes=num_classes)
    path = tmp_path / "model.pt"
    torch.save({key: trained.state_dict()}, path)
    return path, trained


def test_backbone_tensors_arrive_from_the_checkpoint(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path)

    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    assert torch.equal(backbone.model.state_dict()["conv1.weight"], trained.state_dict()["conv1.weight"])


def test_the_classifier_is_stashed_not_loaded(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path)

    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    assert backbone._carried_classifier is not None
    assert torch.equal(backbone._carried_classifier["fc.weight"], trained.state_dict()["fc.weight"])


def test_the_ema_branch_wins_when_present(tmp_path: Path) -> None:
    trained = timm.create_model(MODEL, pretrained=False, num_classes=3)
    ema = timm.create_model(MODEL, pretrained=False, num_classes=3)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": trained.state_dict(), "state_dict_ema": ema.state_dict()}, path)

    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    assert torch.equal(backbone.model.state_dict()["conv1.weight"], ema.state_dict()["conv1.weight"])


def test_a_foreign_architecture_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"decoder.weight": torch.zeros(2, 2)}}, path)

    with pytest.raises(RuntimeError, match="decoder"):
        TimmBackbone(model_name=MODEL, checkpoint_path=path)


def test_strict_false_still_refuses_a_checkpoint_that_matches_nothing(tmp_path: Path) -> None:
    """A user's weight file must never no-op silently."""
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"decoder.weight": torch.zeros(2, 2)}}, path)

    with pytest.raises(ValueError, match="no weights"):
        TimmBackbone(model_name=MODEL, checkpoint_path=path, strict=False)


def test_equal_class_counts_transplant_the_whole_classifier(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path, num_classes=3)
    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    head = backbone.native_head("features", backbone.feature_dim("features"), 3)

    assert head is not None
    assert not isinstance(head, ExpandedHead)
    assert torch.equal(dict(head.named_parameters())["weight"], trained.state_dict()["fc.weight"])


def test_growing_the_class_space_transplants_base_and_leaves_novel_fresh(tmp_path: Path) -> None:
    path, trained = full_model_file(tmp_path, num_classes=3)
    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    head = backbone.native_head("features", backbone.feature_dim("features"), 4)

    assert isinstance(head, ExpandedHead)
    assert torch.equal(dict(head.base.named_parameters())["weight"], trained.state_dict()["fc.weight"])
    novel_weight = dict(head.novel.named_parameters())["weight"]
    assert novel_weight.shape[0] == 1
    assert not torch.equal(novel_weight, trained.state_dict()["fc.weight"][:1])


def test_narrowing_the_class_space_is_refused(tmp_path: Path) -> None:
    path, _ = full_model_file(tmp_path, num_classes=3)
    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    with pytest.raises(ValueError, match="mapping"):
        backbone.native_head("features", backbone.feature_dim("features"), 2)


def test_a_foreign_feature_space_is_refused(tmp_path: Path) -> None:
    path, _ = full_model_file(tmp_path, num_classes=3)
    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    with pytest.raises(ValueError, match="feature"):
        backbone.native_head("features", backbone.feature_dim("features") + 1, 3)


def test_without_a_checkpoint_the_native_head_stays_fresh() -> None:
    backbone = TimmBackbone(model_name=MODEL, pretrained=False)

    head = backbone.native_head("features", backbone.feature_dim("features"), 3)

    assert head is not None
    assert not isinstance(head, ExpandedHead)


def test_transplanted_rows_stay_trainable_until_freeze_says_otherwise(tmp_path: Path) -> None:
    """``no_grad`` at copy time is copy mechanics, not freezing — freezing stays the callback's job."""
    path, _ = full_model_file(tmp_path, num_classes=3)
    backbone = TimmBackbone(model_name=MODEL, checkpoint_path=path)

    head = backbone.native_head("features", backbone.feature_dim("features"), 4)

    assert head is not None
    assert all(parameter.requires_grad for parameter in head.parameters())
