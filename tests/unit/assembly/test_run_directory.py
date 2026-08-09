"""Everything a run writes lands under ``run.directory``, and config is what says so.

Lightning would answer this from the *logger* — right for one that writes files,
wrong for a tracker that uploads: with ClearML a run's weights went to
``runs/<project>/<name>/<name>/<task id>/checkpoints``, its identity spelled twice
plus a hash, three levels below everything else it produced. The correction is an
interpolation on each declaration rather than a step in assembly, so these tests
guard the declarations — a shipped group that forgets one is the whole failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from src.callbacks import callback_registry

CONFIGS = Path("configs")
ROOTED = "${run.directory}"


def documents() -> list[tuple[Path, dict[str, Any]]]:
    """Every shipped config file, with its path for the failure message."""
    loaded = []
    for path in sorted(CONFIGS.rglob("*.yaml")):
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            loaded.append((path, content))
    return loaded


def savers() -> list[tuple[Path, dict[str, Any]]]:
    """Every checkpoint declared in a shipped config, found through the registry.

    By what the name *builds* rather than by how it reads, so a saver registered
    under some future name is covered the day it is added rather than the day
    someone remembers to extend a list of names here.
    """
    found = []
    for path, document in documents():
        for entry in document.get("callbacks") or []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if name is None:
                continue
            built = callback_registry.get(name)
            if isinstance(built, type) and issubclass(built, ModelCheckpoint):
                found.append((path, entry))
    return found


def test_the_shipped_configs_declare_savers_at_all() -> None:
    """A guard on a guard: were the declarations renamed away, the test below would
    pass by finding nothing, and the weights would go back under the tracker's subtree.
    """
    assert len(savers()) >= 3


@pytest.mark.parametrize("path,entry", savers(), ids=lambda value: str(value)[:40])
def test_every_shipped_saver_writes_into_the_runs_own_directory(path: Path, entry: dict[str, Any]) -> None:
    """One run, one directory — the same one `export` writes into and the job log sits in."""
    assert entry.get("dirpath", "").startswith(ROOTED), f"{path}: {entry.get('name')} declares no rooted dirpath"


def test_the_trainer_group_roots_lightnings_own_default_there() -> None:
    """What nothing else claims — `lightning_logs/`, a checkpoint with no `dirpath` —
    still belongs to the run. Left out, Lightning falls back to the working directory,
    and Hydra does not chdir, so that is wherever the command was typed.
    """
    trainer = yaml.safe_load((CONFIGS / "trainer" / "default.yaml").read_text(encoding="utf-8"))

    assert trainer["default_root_dir"] == ROOTED


def test_the_profiler_preset_writes_its_report_there_too() -> None:
    """And with a file name, because that is a switch rather than a name: a profiler
    writes a file only when it has both, and with a directory alone the report goes
    to the job log instead — three hundred lines of it.
    """
    preset = yaml.safe_load((CONFIGS / "trainer" / "profile.yaml").read_text(encoding="utf-8"))

    assert preset["profiler"]["dirpath"] == ROOTED
    assert preset["profiler"]["filename"]


def test_the_interpolation_resolves_to_the_directory_rather_than_reading_like_it() -> None:
    """The tests above match a string; this one shows the string is a live reference.

    A key renamed at the root would leave every declaration above looking correct
    and resolving to nothing.
    """
    declared = OmegaConf.create(
        {
            "run": {"directory": "runs/pets/one"},
            "callbacks": [{"name": "checkpoint", "dirpath": f"{ROOTED}/checkpoints"}],
            "trainer": {"profiler": {"name": "simple", "dirpath": ROOTED}},
        }
    )

    resolved = OmegaConf.to_container(declared, resolve=True)

    assert isinstance(resolved, dict)
    assert resolved["callbacks"][0]["dirpath"] == "runs/pets/one/checkpoints"
    assert resolved["trainer"]["profiler"]["dirpath"] == "runs/pets/one"
