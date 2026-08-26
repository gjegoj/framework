"""Command-line entry point: Hydra composes, assembly builds, the trainer runs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax

from src.assembly import assemble, run
from src.config import load_config

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[1] / "configs")
"""Where the YAML groups live, as an absolute path — the one form every entry point shares."""


@hydra.main(version_base=None, config_path=CONFIG_DIRECTORY, config_name="config")
def main(composed: DictConfig) -> None:
    """Compose config, assemble the experiment, and run it.

    Hydra is confined to this module — it composes the YAML groups and applies CLI
    overrides, and everything below receives one validated ``ExperimentConfig``::

        uv run main.py experiment=examples/classification
        uv run main.py experiment=examples/classification lr=3e-4 trainer.max_epochs=50
        uv run main.py experiment=examples/classification run.train=false +run.checkpoint_path=runs/best.ckpt
        uv run main.py experiment=examples/classification +run.resume_path=runs/last.ckpt

    A key the composed config already declares is overridden by name; one it does not —
    ``run.checkpoint_path``, ``run.resume_path`` — is *added* with Hydra's ``+``.
    """
    silence_third_party_notices()
    raw = cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True))
    show_config(raw)
    config = load_config(raw)
    run(assemble(config), config)


def silence_third_party_notices() -> None:
    """Drop the notices about how one library calls another, and only those."""
    warnings.filterwarnings("ignore", message=r".*LeafSpec.*is deprecated", category=FutureWarning)
    warnings.filterwarnings(
        "ignore", message=r".*Precision bf16-mixed is not supported by the model summary.*", category=FutureWarning
    )


def show_config(resolved: Mapping[str, Any]) -> None:
    """Show the run what it was given, in the language a config is written in."""
    rendered = yaml.safe_dump(dict(resolved), default_flow_style=None, sort_keys=False, allow_unicode=True)
    rprint(
        Panel(
            Syntax(rendered, "yaml", theme="perldoc", background_color="default"), title="Configuration", expand=False
        )
    )


if __name__ == "__main__":
    main()
