"""A checkpoint carries the weights of the model that ships, whatever scaffolding wrote it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import pytest
import torch
from torch import Tensor, nn

from src.assembly.checkpoints import load_weights, shipped_weights
from src.core import Batch, Loss, Model, Prediction, StepResult
from src.models import DistilledModel
from src.models.adapters import LoraAdapters

if TYPE_CHECKING:
    from pathlib import Path


class Tiny(Model):
    """One linear layer — the smallest model whose weights can be told apart."""

    def __init__(self, fill: float) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        with torch.no_grad():
            self.net.weight.fill_(fill)

    def step(self, batch: Batch) -> StepResult:
        return StepResult(loss=Loss.part("ce", torch.tensor(0.0)), prediction=self.predict(batch), targets={})

    def predict(self, batch: Batch) -> Prediction:
        logits = self.net(batch.inputs["image"])
        return Prediction(outputs={"label": logits}, logits={"label": logits})


class Learned(nn.Module):
    """A criterion with state of its own — training scaffolding, not shipped weights."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.part("kl", (logits - target).abs().mean() / self.temperature)


class Holder(L.LightningModule):
    """Stands in for `TrainingModule`: a run's checkpoint is this module's `state_dict`."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


def written(path: Path, model: nn.Module) -> str:
    torch.save({"state_dict": Holder(model).state_dict()}, path)
    return str(path)


def distilled(student: Model) -> DistilledModel:
    return DistilledModel(
        student=student,
        teachers=[Tiny(9.0)],
        criterion=Learned(),  # type: ignore[arg-type]
    )


def test_a_distilled_runs_checkpoint_loads_into_a_plain_model(tmp_path: Path) -> None:
    """Exporting or testing a trained student later must not require re-declaring its teachers."""
    source = written(tmp_path / "distilled.ckpt", distilled(Tiny(1.0)))
    plain = Tiny(0.0)

    load_weights(plain, source)

    assert torch.allclose(plain.net.weight, torch.full_like(plain.net.weight, 1.0))


def test_a_plain_runs_checkpoint_loads_into_a_distilled_model(tmp_path: Path) -> None:
    """Warm-starting distillation from an already trained student is the ordinary way to reach for it."""
    source = written(tmp_path / "plain.ckpt", Tiny(1.0))
    model = distilled(Tiny(0.0))

    load_weights(model.student, source)

    student = model.student
    assert isinstance(student, Tiny)
    assert torch.allclose(student.net.weight, torch.full_like(student.net.weight, 1.0))


def test_a_criterions_own_state_is_not_mistaken_for_shipped_weights(tmp_path: Path) -> None:
    """What ships is the student; a criterion that learned something is scaffolding the run wore."""
    source = written(tmp_path / "distilled.ckpt", distilled(Tiny(1.0)))

    weights = shipped_weights(source)

    assert set(weights) == set(Tiny(0.0).state_dict())


def test_a_file_that_is_not_our_checkpoint_names_the_field_that_takes_it(tmp_path: Path) -> None:
    """A backbone's arrived weights are a different kind of file, and the message says where they go."""
    foreign = tmp_path / "backbone.pth"
    torch.save({"net.weight": torch.zeros(2, 4)}, foreign)

    with pytest.raises(ValueError, match="model.checkpoint_path"):
        load_weights(Tiny(0.0), str(foreign))


def test_a_plain_checkpoint_loads_beneath_freshly_added_adapters(tmp_path: Path) -> None:
    """Warm-starting LoRA from a plain run's weights is the ordinary way to reach for it.

    The base loads under the adapters and the deltas keep their zero start, so the
    adapted model computes exactly what the checkpoint's weights say until training
    moves the delta."""
    source = written(tmp_path / "plain.ckpt", Tiny(1.0))
    adapted = Tiny(0.0)
    LoraAdapters(target_modules=["net"], rank=2)(adapted)

    load_weights(adapted, source)

    weights = shipped_weights(source)
    grafted = adapted.state_dict()["net.base_layer.weight"]
    assert torch.allclose(grafted, torch.full_like(grafted, 1.0))
    x = torch.randn(5, 4)
    assert torch.allclose(adapted.net(x), torch.nn.functional.linear(x, weights["net.weight"], weights["net.bias"]))


def test_an_adapted_runs_checkpoint_still_loads_strictly_into_an_adapted_model(tmp_path: Path) -> None:
    """The evaluate-a-lora-checkpoint workflow: adapted keys fit an adapted model as-is."""
    trained = Tiny(1.0)
    LoraAdapters(target_modules=["net"], rank=2)(trained)
    source = written(tmp_path / "lora.ckpt", trained)
    fresh = Tiny(0.0)
    LoraAdapters(target_modules=["net"], rank=2)(fresh)

    load_weights(fresh, source)

    grafted = fresh.state_dict()["net.base_layer.weight"]
    assert torch.allclose(grafted, torch.full_like(grafted, 1.0))


def test_a_checkpoint_of_a_different_architecture_is_refused_even_beneath_adapters(tmp_path: Path) -> None:
    """The graft fixes exactly one mismatch — the adapters' rename — and nothing else."""
    source = written(tmp_path / "foreign.ckpt", nn.Linear(8, 8))
    adapted = Tiny(0.0)
    LoraAdapters(target_modules=["net"], rank=2)(adapted)

    with pytest.raises(ValueError, match="beneath the adapters"):
        load_weights(adapted, source)


def test_a_size_mismatch_is_refused_by_name_even_beneath_adapters(tmp_path: Path) -> None:
    """A checkpoint whose layer grew a different width dies with the framework's own
    refusal, not a raw size-mismatch traceback: the graft renames keys, and a shape
    that disagrees after the rename is the architecture disagreeing."""

    class Wide(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 20)

    source = written(tmp_path / "wide.ckpt", Wide())
    adapted = Tiny(0.0)
    LoraAdapters(target_modules=["net"], rank=2)(adapted)

    with pytest.raises(ValueError, match="beneath the adapters"):
        load_weights(adapted, source)


def test_a_checkpoint_that_does_not_fit_names_the_model_it_did_not_fit(tmp_path: Path) -> None:
    """Two shapes are refused here — an adapted run's and a mismatched teacher's — so the message names both."""
    source = written(tmp_path / "plain.ckpt", Tiny(1.0))

    with pytest.raises(ValueError, match="Linear"):
        load_weights(nn.Linear(8, 8), source)
