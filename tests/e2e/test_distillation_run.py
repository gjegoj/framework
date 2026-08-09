"""End-to-end: a distilled run trains, and what it leaves behind is about the student alone."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from src.assembly import assemble, run
from src.assembly.checkpoints import load_weights
from src.models import DistilledModel
from tests.support.configs import disk_config

if TYPE_CHECKING:
    from pathlib import Path

TEACHER = {"name": "timm", "model_name": "resnet34", "pretrained": False}

LOADER = {"batch_size": 2, "drop_last": True}
"""Two batches of two, so a step sees a full one — a distilled step has to actually run."""

IDLE = {"train": False, "test": False}
"""Neither phase, for the tests that only want the assembled experiment."""


def distilling(root: Path, **sections: Any) -> Any:
    """The base run at a full batch, plus whichever section a test is about."""
    return disk_config(root, loader=LOADER, **({"run": IDLE} | sections))


@pytest.mark.e2e
@pytest.mark.slow
def test_a_distilled_run_ships_an_artifact_the_size_of_its_student(dataset_root: Path) -> None:
    """The teachers are held off the module tree, and a traced graph carries whatever is on it.

    Measured: a teacher registered as a submodule lands in the artifact — 292
    parameters against the student's 10 on a toy pair — while one held in a plain
    list does not. This is what keeps that true through a whole run.
    """
    config = distilling(
        dataset_root,
        distillation={
            "teachers": [{"model": TEACHER}],
            "loss": {"name": "kl_divergence", "temperature": 2.0, "weight": 0.5},
        },
        export=[{"name": "torchscript"}],
        run={"directory": str(dataset_root / "run"), "test": True},
    )

    experiment = assemble(config)
    run(experiment, config)

    artifact = dataset_root / "run" / "export" / "model.pt"
    assert artifact.is_file()
    assert isinstance(experiment.module.model, DistilledModel)
    loaded = torch.jit.load(str(artifact))
    shipped = sum(parameter.numel() for parameter in loaded.parameters())
    student = sum(parameter.numel() for parameter in experiment.module.model.student.parameters())
    assert shipped == student


@pytest.mark.e2e
@pytest.mark.slow
def test_a_distilled_runs_checkpoint_opens_a_plain_run(dataset_root: Path, tmp_path: Path) -> None:
    """Testing or exporting a trained student later must not require re-declaring its teachers.

    A distilled run keys its checkpoint `model.student.…`, so without a reader
    that knows what ships, the ordinary follow-up run — `checkpoint_path` with
    `train: false` — would have to rebuild teachers it will never ask anything of.
    """
    teaching = distilling(
        dataset_root,
        distillation={"teachers": [{"model": TEACHER}]},
        run={"directory": str(dataset_root / "run"), "train": True, "test": False},
    )
    experiment = assemble(teaching)
    run(experiment, teaching)
    checkpoint = tmp_path / "distilled.ckpt"
    experiment.trainer.save_checkpoint(checkpoint)
    assert isinstance(experiment.module.model, DistilledModel)
    trained = experiment.module.model.student.state_dict()

    plain = assemble(distilling(dataset_root))
    load_weights(plain.module.model, str(checkpoint))

    reopened = plain.module.model.state_dict()
    assert all(torch.allclose(reopened[name], value) for name, value in trained.items())
