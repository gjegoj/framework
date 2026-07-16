"""Segmentation criteria: Dice wrapper and verbatim kwargs forwarding to smp."""

from __future__ import annotations

import torch

from src.losses.registry import criteria


class TestDiceCriterion:
    def test_registered(self) -> None:
        assert "dice" in criteria

    def test_extra_kwargs_forward_to_smp_loss(self) -> None:
        """Unlisted smp.DiceLoss params (smooth/log_loss/eps/...) are configurable from YAML."""
        import segmentation_models_pytorch as smp

        criterion = criteria.create("dice", smooth=1.0, log_loss=True, eps=1e-5)
        wrapped = criterion._loss  # noqa: SLF001 — pinning the forwarding contract
        assert isinstance(wrapped, smp.losses.DiceLoss)
        assert wrapped.smooth == 1.0
        assert wrapped.log_loss is True
        assert wrapped.eps == 1e-5

    def test_multiclass_loss_computes(self) -> None:
        criterion = criteria.create("dice", smooth=1.0)
        logits = torch.randn(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))
        result = criterion(logits, target)
        assert result.total.ndim == 0
        assert "dice" in result.components
