"""``TimmBackbone``: any timm model as a pooled-feature backbone."""

from __future__ import annotations

import pytest
import torch

from src.core import (
    Batch,
    DataProfile,
    Features,
    Objective,
    OutputTopology,
    Stream,
    TargetFacts,
    Task,
)
from src.models import CompositeModel, TimmBackbone
from src.models.registry import backbone_registry
from src.tasks import build_task_components
from tests.support.narrowing import tensor


@pytest.fixture(scope="module")
def backbone() -> TimmBackbone:
    return TimmBackbone(model_name="resnet18", pretrained=False)


def test_exposes_pooled_features_under_the_default_stream(backbone: TimmBackbone) -> None:
    features = backbone({"image": torch.randn(2, 3, 64, 64)})

    assert isinstance(features, Features)
    assert features[Stream.FEATURES].shape == (2, 512)


def test_feature_dim_reports_the_model_width(backbone: TimmBackbone) -> None:
    assert backbone.feature_dim(Stream.FEATURES) == 512


def test_unknown_stream_is_rejected_by_name(backbone: TimmBackbone) -> None:
    with pytest.raises(LookupError, match=Stream.FEATURES):
        backbone.feature_dim("decoder")


def test_registered_under_the_timm_key() -> None:
    created = backbone_registry.create("timm", model_name="resnet18", pretrained=False)

    assert isinstance(created, TimmBackbone)


def test_kwargs_forward_to_timm_create_model() -> None:
    grayscale = TimmBackbone(model_name="resnet18", pretrained=False, in_chans=1)

    features = grayscale({"image": torch.randn(2, 1, 64, 64)})

    assert features[Stream.FEATURES].shape == (2, 512)


def test_input_name_selects_the_batch_input() -> None:
    named = TimmBackbone(model_name="resnet18", pretrained=False, input_name="frame")

    features = named({"frame": torch.randn(1, 3, 64, 64)})

    assert Stream.FEATURES in features


def test_native_head_is_timms_classifier(backbone: TimmBackbone) -> None:
    head = backbone.native_head(Stream.FEATURES, in_features=512, out_features=3)

    assert head is not None
    assert head(torch.randn(2, 512)).shape == (2, 3)


def test_native_head_is_none_for_other_streams(backbone: TimmBackbone) -> None:
    assert backbone.native_head(Stream.DECODER, in_features=512, out_features=3) is None


def test_heads_are_sized_from_the_real_model(backbone: TimmBackbone) -> None:
    profile = DataProfile()
    profile.record("label", TargetFacts(num_classes=3))
    task = Task(name="label", output_topology=OutputTopology.GLOBAL, objective=Objective.MULTICLASS, metrics={})

    components = build_task_components(task, profile, backbone)
    model = CompositeModel(backbone=backbone, components={"label": components})

    prediction = model.predict(Batch(inputs={"image": torch.randn(2, 3, 64, 64)}, targets={}))

    assert tensor(prediction.outputs["label"]).shape == (2, 3)
