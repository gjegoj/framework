"""What a model calls itself — the one identity a run is filed under in a tracker."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from src.core import Backbone, Batch, Features, Loss, Model, Prediction, StepResult
from src.core.taxonomy import Stream
from src.models import (
    CompositeModel,
    DistilledModel,
    LinearHead,
    MultiEncoderBackbone,
    MultiViewBackbone,
    SmpBackbone,
    TaskComponents,
    TimmBackbone,
)


class Nameless(Backbone):
    """A backbone that overrides nothing, to read the port's own default."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> Features:
        return Features(streams={Stream.FEATURES: inputs["image"].flatten(1)})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: 4}


class Vendor(Model):
    """A family owning its own head and loss, so it has no backbone to ask."""

    def step(self, batch: Batch) -> StepResult:
        return StepResult(loss=Loss.part("ce", torch.tensor(0.0)), prediction=self.predict(batch), targets={})

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": torch.zeros(1)})


def composed(backbone: Backbone) -> CompositeModel:
    return CompositeModel(
        backbone=backbone,
        components={
            "label": TaskComponents(
                head=LinearHead(backbone.feature_dim(Stream.FEATURES), 2),
                criterion=nn.Identity(),  # type: ignore[arg-type]
                activation=lambda logits: logits,
                target_adapter=None,
            )
        },
    )


def test_timm_answers_with_the_name_it_normalised_not_the_one_declared() -> None:
    """Measured: timm resolves `resnet18.a1_in1k` to `resnet18`, so its own answer beats the config's."""
    assert TimmBackbone("resnet18.a1_in1k", pretrained=False).architecture == "resnet18"


def test_smp_answers_with_the_declaration_because_its_own_name_abbreviates() -> None:
    """Measured: smp calls a Unet on a ResNet-34 `u-resnet34`, which is not what anyone filters by."""
    assert SmpBackbone(arch="unet", encoder_name="resnet34", pretrained=False).architecture == "unet-resnet34"


def test_a_composite_backbone_joins_what_it_holds() -> None:
    """This is the case the reference dropped: a config interpolation has no key to reach here."""
    encoders = {
        "image": TimmBackbone("resnet18", pretrained=False),
        "other": TimmBackbone("resnet34", pretrained=False),
    }

    joined = MultiEncoderBackbone(encoders=encoders, embedding_dim=8).architecture

    assert joined == "resnet18+resnet34"


def test_reading_one_encoder_over_several_views_is_still_that_encoder() -> None:
    """Views are a way of reading, not another model, so a run stays filed under the encoder."""
    assert MultiViewBackbone(inner=TimmBackbone("resnet18", pretrained=False)).architecture == "resnet18"


def test_a_backbone_that_names_nothing_falls_back_to_its_class() -> None:
    """The default has to say something: a run with no name in the list cannot be found in it."""
    assert Nameless().architecture == "Nameless"


def test_a_composed_model_answers_from_its_backbone() -> None:
    """What a composed run is, is what encodes for it — the heads follow from the tasks."""
    assert composed(TimmBackbone("resnet18", pretrained=False)).architecture == "resnet18"


def test_a_distilled_run_is_filed_under_the_student() -> None:
    """The teachers are scaffolding; a run found by their name would be found by the wrong one."""
    student = composed(TimmBackbone("resnet18", pretrained=False))
    teacher = composed(TimmBackbone("resnet34", pretrained=False))

    distilled = DistilledModel(student=student, teachers=[teacher], criterion=nn.Identity())  # type: ignore[arg-type]

    assert distilled.architecture == "resnet18"


def test_a_family_with_no_backbone_answers_for_itself() -> None:
    """A vendor model owns its own head and loss, and its class name is the only honest answer."""
    assert Vendor().architecture == "Vendor"
