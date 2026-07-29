"""CompleteModel port: contract surface, registry, single-kind-namespace guard."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch
from torch import nn

from src.core.entities import LossResult
from src.models.complete import CompleteModel
from src.models.registry import backbones, complete_models, register_complete_model


class _StubModel(CompleteModel[torch.Tensor, torch.Tensor]):
    family: ClassVar[str] = "stub"

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return cast("torch.Tensor", self.linear(batch["inputs"]))

    def training_loss(self, batch: dict[str, Any]) -> LossResult:
        total = self.forward(batch).sum()
        return LossResult(total=total, components={"stub": total.detach()})

    def evaluation_loss(self, batch: dict[str, Any], output: torch.Tensor) -> LossResult:
        return LossResult(total=output.sum(), components={})

    def predictions(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def targets(self, batch: dict[str, Any]) -> torch.Tensor:
        return cast("torch.Tensor", batch["target"])


class TestCompleteModelContract:
    def test_abstract_methods_required(self) -> None:
        class Incomplete(CompleteModel[torch.Tensor, torch.Tensor]):
            family: ClassVar[str] = "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_prepare_batch_defaults_to_identity(self) -> None:
        batch = {"inputs": torch.randn(1, 2)}
        assert _StubModel().prepare_batch(batch) is batch

    def test_is_an_nn_module(self) -> None:
        assert isinstance(_StubModel(), nn.Module)


class TestKindNamespace:
    def test_kind_colliding_with_backbone_kind_rejected(self) -> None:
        assert "timm" in backbones  # precondition: the backbone kind exists
        with pytest.raises(ValueError, match="single .?kind.? namespace"):
            register_complete_model("timm")(_StubModel)

    def test_registered_kind_creates_instances(self) -> None:
        register_complete_model("stub-for-test")(_StubModel)
        try:
            assert isinstance(complete_models.create("stub-for-test"), _StubModel)
        finally:
            complete_models._factories.pop("stub-for-test")  # test-only cleanup, registry has no unregister
