"""End-to-end: a finished run leaves a TorchScript file that still is the model.

Every layer participates — config, data, model assembly, Lightning fit, weight
restoration, the export phase and its verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.assembly import assemble, run
from src.core import Batch
from tests.support.configs import disk_config
from tests.support.narrowing import tensor


@pytest.mark.e2e
def test_a_finished_run_leaves_a_loadable_artifact_that_matches_the_model(dataset_root: Path, tmp_path: Path) -> None:
    """An export nobody can load, or that computes something else, is not a deliverable."""
    config = disk_config(
        dataset_root,
        export=[{"name": "torchscript"}],
        run={"directory": str(tmp_path / "run"), "test": False},
    )

    experiment = assemble(config)
    run(experiment, config)

    artifact = tmp_path / "run" / "export" / "model.pt"
    assert artifact.is_file()

    loaded = torch.jit.load(str(artifact))
    loaded.eval()
    probe = torch.randn(1, 3, 16, 16)
    model = experiment.module.model
    model.eval()
    model.cpu()
    with torch.no_grad():
        expected = tensor(model.predict(Batch(inputs={"image": probe}, targets={})).outputs["label"])
        written = loaded(probe)

    # The run has one task, so the artifact hands back the tensor itself; indexing it
    # here would compare a row against the whole output and pass by broadcasting.
    assert written.shape == expected.shape
    assert torch.allclose(written, expected, atol=1e-5)
