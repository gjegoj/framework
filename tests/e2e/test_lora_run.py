"""End-to-end: a LoRA run trains, and what it ships carries no trace of the technique."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.assembly import assemble, run
from tests.support.configs import disk_config

VIT = {"name": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 32}
"""A ViT carries the module names LoRA targets; 32 matches what the transforms resize to."""


@pytest.mark.e2e
@pytest.mark.slow
def test_a_lora_run_ships_an_artifact_with_no_adapters_in_it(dataset_root: Path, tmp_path: Path) -> None:
    """The technique is a training-time reparameterization; a deployment should not have to know it happened."""
    config = disk_config(
        dataset_root,
        image_size=[32, 32],
        model=VIT,
        adapters={"name": "lora", "target_modules": ["qkv", "proj"], "rank": 4},
        loader={"batch_size": 2, "drop_last": True},
        export=[{"name": "torchscript"}],
        run={"directory": str(tmp_path / "run"), "test": True},
    )

    experiment = assemble(config)
    adapted = [name for name in experiment.module.state_dict() if "lora_" in name]
    assert adapted, "the run must actually be adapted, or this test proves nothing"

    run(experiment, config)

    artifact = tmp_path / "run" / "export" / "model.pt"
    assert artifact.is_file()
    loaded = torch.jit.load(str(artifact))
    names = [name for name, _ in loaded.named_parameters()]
    assert not any("lora_" in name or "base_layer" in name for name in names)
