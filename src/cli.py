"""The command line: Hydra composes the declaration, the root wires it, the trainer runs it.

Hydra and OmegaConf are confined to this module — they compose the YAML groups and apply the
overrides, and everything below receives one validated ``ExperimentConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax

from src.build import build
from src.config import load_config
from src.console import console
from src.experiment import run

CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[1] / "configs")
"""Where the YAML groups live, as an absolute path — the one form every entry point shares."""


@hydra.main(version_base=None, config_path=CONFIG_DIRECTORY, config_name="config")
def main(composed: DictConfig) -> None:
    """Compose the declaration, build the run, and start it::

        uv run main.py experiment=examples/classification
        uv run main.py experiment=examples/classification lr=3e-4 epochs=50 tracker=clearml
        uv run main.py experiment=examples/classification run.train=false +run.checkpoint_path=runs/best.ckpt

    A key the composed config already declares is overridden by name; one it does not —
    ``run.checkpoint_path``, ``run.resume_path`` — is *added* with Hydra's ``+``.
    """
    show(composed)
    # `throw_on_missing`, because the root config marks `data.source` as `???`: without it the marker
    # degrades into that literal string, and a run that forgot its source is told its file format is
    # wrong. Shown first, so the panel is there to read when the refusal comes.
    resolved = cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True, throw_on_missing=True))
    run(build(load_config(resolved)))


def show(composed: DictConfig) -> None:
    """Show the run what it was given, in the language a declaration is written in."""
    rendered = OmegaConf.to_yaml(composed, resolve=True)
    console().print(
        Panel(Syntax(rendered, "yaml", theme="perldoc", background_color="default"), title="Run", expand=False)
    )


if __name__ == "__main__":
    main()
