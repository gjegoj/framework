"""Target adapters: loss/metric tensor views per objective (src.tasks.adapters)."""

from __future__ import annotations

import torch


class TestTargetAdapters:
    def test_binary_codec_shapes(self) -> None:
        from src.tasks.adapters import BinaryTargetAdapter

        view = BinaryTargetAdapter().adapt(torch.tensor([0, 1, 1, 0]))
        assert view.loss.shape == (4, 1) and view.loss.dtype == torch.float
        assert view.metric.shape == (4, 1) and view.metric.dtype == torch.long

    def test_multilabel_codec_shapes(self) -> None:
        from src.tasks.adapters import MultilabelTargetAdapter

        target = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)
        view = MultilabelTargetAdapter().adapt(target)
        assert view.loss.dtype == torch.float
        assert view.metric.dtype == torch.long

    def test_continuous_codec_shapes(self) -> None:
        from src.tasks.adapters import ContinuousTargetAdapter

        view = ContinuousTargetAdapter().adapt(torch.tensor([1.5, 2.3, 0.1]))
        assert view.loss.shape == (3, 1) and view.loss.dtype == torch.float
        assert view.metric.shape == (3, 1) and view.metric.dtype == torch.float
