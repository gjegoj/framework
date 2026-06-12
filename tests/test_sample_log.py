"""Unit tests for SampleLogCallback (producer → annotators → renderer → log_html)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.callbacks.sample_log import SampleLogCallback, _denormalize_to_uint8
from src.core.constants import IMAGENET_MEAN, IMAGENET_STD
from src.core.entities import Batch, StepOutput, TaskStepView, is_training_step_output
from src.core.keys import IMAGE
from src.core.ports import PlotLogger
from src.models.assembly import build_composite_model
from src.models.backbones import EmbeddingBackbone
from src.tasks import classification
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder
from src.visualization.renderer import Renderer
from tests.test_metrics import FakePlotLogger


def _lit() -> LitModule:
    task = replace(classification("species", num_classes=3), class_names=["cat", "cow", "dog"])
    model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"species": task.head_spec})
    return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))


def _batch() -> Batch:
    return Batch(inputs={IMAGE: torch.randn(4, 3, 16, 16)}, targets={"species": torch.tensor([0, 1, 2, 0])})


def _outputs() -> StepOutput:
    output: StepOutput = {
        "loss": torch.tensor(1.0),
        "task_views": {
            "species": TaskStepView(
                preds=torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]),
                metric_target=torch.tensor([0, 1, 2, 0]),
            )
        },
    }
    assert is_training_step_output(output)
    return output


def _trainer(logger: Any, epoch: int = 0) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.logger = logger
    return trainer


class _RecordingRenderer(Renderer):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def render(self, samples: list[Any], title: str) -> str:
        self.calls.append({"samples": samples, "title": title})
        return "<html>fake</html>"


class TestFakePlotLoggerContract:
    def test_fake_plot_logger_satisfies_plot_logger(self) -> None:
        assert isinstance(FakePlotLogger(), PlotLogger)


class TestSampleLogCallbackGating:
    def test_should_log_on_matching_epoch_and_batch(self) -> None:
        cb = SampleLogCallback(every_n_epochs=2, batch_index=1)
        assert cb._should_log(0, 1) is True
        assert cb._should_log(2, 1) is True

    def test_should_not_log_wrong_batch(self) -> None:
        cb = SampleLogCallback(batch_index=0)
        assert cb._should_log(0, 1) is False

    def test_should_not_log_wrong_epoch(self) -> None:
        cb = SampleLogCallback(every_n_epochs=2, batch_index=0)
        assert cb._should_log(1, 0) is False

    def test_invalid_num_images_raises(self) -> None:
        with pytest.raises(ValueError, match="num_images"):
            SampleLogCallback(num_images=0)


class TestDenormalize:
    def test_output_shape_and_dtype(self) -> None:
        tensor = torch.zeros(2, 3, 16, 16)
        mean = torch.tensor(list(IMAGENET_MEAN))
        std = torch.tensor(list(IMAGENET_STD))
        result = _denormalize_to_uint8(tensor, num_images=2, mean=mean, std=std)
        assert result.shape == (2, 16, 16, 3)
        assert result.dtype == torch.uint8


class TestSampleLogCallback:
    def test_logs_html_on_should_log_batch(self) -> None:
        fake_logger = FakePlotLogger()
        renderer = _RecordingRenderer()
        cb = SampleLogCallback(
            num_images=2, batch_index=0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), renderer=renderer
        )
        cb.on_validation_batch_end(_trainer(fake_logger), _lit(), _outputs(), _batch(), batch_idx=0)

        assert len(renderer.calls) == 1
        assert len(renderer.calls[0]["samples"]) == 2
        sample = renderer.calls[0]["samples"][0]
        assert "species_gt" in sample.fields and "species_pred" in sample.fields
        assert len(fake_logger.html_calls) == 1
        assert fake_logger.html_calls[0]["title"] == "samples/val"
        assert fake_logger.html_calls[0]["html"] == "<html>fake</html>"

    def test_no_op_without_plot_logger(self) -> None:
        renderer = _RecordingRenderer()
        cb = SampleLogCallback(num_images=2, renderer=renderer)
        cb.on_validation_batch_end(_trainer(MagicMock()), _lit(), _outputs(), _batch(), batch_idx=0)
        assert renderer.calls == []

    def test_skips_wrong_batch_index(self) -> None:
        renderer = _RecordingRenderer()
        cb = SampleLogCallback(num_images=2, batch_index=0, renderer=renderer)
        cb.on_validation_batch_end(_trainer(FakePlotLogger()), _lit(), _outputs(), _batch(), batch_idx=3)
        assert renderer.calls == []

    def test_logs_html_on_test_batch_end(self) -> None:
        fake_logger = FakePlotLogger()
        cb = SampleLogCallback(num_images=2, batch_index=0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        cb.on_test_batch_end(_trainer(fake_logger), _lit(), _outputs(), _batch(), batch_idx=0)
        assert len(fake_logger.html_calls) == 1
        assert fake_logger.html_calls[0]["title"] == "samples/test"

    def test_no_op_for_embedding_input(self) -> None:
        fake_logger = FakePlotLogger()
        renderer = _RecordingRenderer()
        cb = SampleLogCallback(num_images=2, renderer=renderer)
        batch = Batch(inputs={"embedding": torch.randn(4, 16)}, targets={"species": torch.tensor([0, 1, 2, 0])})
        cb.on_validation_batch_end(_trainer(fake_logger), _lit(), _outputs(), batch, batch_idx=0)
        assert renderer.calls == []
        assert fake_logger.html_calls == []

    def test_does_not_recompute_activation_or_codec(self) -> None:
        fake_logger = FakePlotLogger()
        cb = SampleLogCallback(num_images=2, batch_index=0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        lit = _lit()
        task = lit.tasks[0]
        with (
            patch.object(task.activation, "__call__", wraps=task.activation.__call__) as mock_activation,
            patch.object(task.codec, "adapt", wraps=task.codec.adapt) as mock_adapt,
        ):
            cb.on_validation_batch_end(_trainer(fake_logger), lit, _outputs(), _batch(), batch_idx=0)
        mock_activation.assert_not_called()
        mock_adapt.assert_not_called()
