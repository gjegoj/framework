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
"""Where the YAML groups live, as an absolute path — the one form every entry point shares.

Hydra resolves a *relative* ``config_path`` against ``task_function.__module__``, so what
it means depends on how the run was started: ``main.py`` at the root imports this module
and leaves it as ``src.cli``, where ``../configs`` becomes the import path ``configs``
and the run dies with "Primary config module 'configs' not found", while
``python -m src.cli`` leaves it as ``__main__`` and the same string resolves against the
file. An absolute path is returned verbatim by ``compute_search_path_dir``, so every
spelling of the command reaches the same directory.
"""


@hydra.main(version_base=None, config_path=CONFIG_DIRECTORY, config_name="config")
def main(composed: DictConfig) -> None:
    """Compose config, assemble the experiment, and run it.

    Hydra is confined to this module — it composes the YAML groups and applies CLI
    overrides, and everything below receives one validated ``ExperimentConfig``, the way
    Lightning stays inside ``training/`` and pydantic inside ``config/``::

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
    """Drop the notices about how one library calls another, and only those.

    Exactly one qualifies today. Lightning keeps a fork of torch's ``tree_flatten``
    so that containers of primitives are not flattened, and that fork builds
    ``LeafSpec()`` — a class torch has since deprecated. Traced, the notice comes
    from ``CombinedLoader.__init__`` while any set of dataloaders is prepared, so
    it arrives on every run whatever the data; and torch itself suppresses the
    same notice where it builds its own leaf singleton, which is the measurement
    that says the fix is one line upstream rather than anything here.

    Everything else a run prints stays. A tip about ``num_workers`` or an unused
    GPU describes *this* run and is answered by changing it, so hiding it would
    hide a choice we made. The category is no shortcut either: Lightning files
    "your val dataloader's sampler has shuffling enabled" and "you defined a
    validation_step but have no val_dataloader" under ``PossibleUserWarning``
    too, and a run that silently skips validation must not be quiet about it.

    Lives with the entry point rather than in ``assemble`` or ``run``: warning
    filters belong to a process, and a notebook that imports those two did not
    ask us to change what its own code is allowed to report.
    """
    warnings.filterwarnings("ignore", message=r".*LeafSpec.*is deprecated", category=FutureWarning)
    warnings.filterwarnings(
        "ignore", message=r".*Precision bf16-mixed is not supported by the model summary.*", category=FutureWarning
    )


def show_config(resolved: Mapping[str, Any]) -> None:
    """Show the run what it was given, in the language a config is written in.

    YAML rather than a rendered tree: a config *is* YAML, so what reaches the
    screen can be pasted straight back into a file.

    Shown before validation on purpose. A config that is refused is the one most
    worth looking at, and the refusal names the offending key without showing
    what surrounded it. It is also the only place a run's substituted values
    appear at all — measured, Hydra archives the composed config with its
    interpolations intact (``lr: ${lr}``, ``mean: ${mean}``), and that archive
    cannot be re-resolved on its own, so what the run actually read is nowhere
    else on disk.

    Takes the resolved mapping rather than the composed config so that this is
    exactly the object validation receives — and so time-based interpolations
    are resolved once, not once for the screen and again for the run.

    ``default_flow_style=None`` keeps a collection of scalars on its own line:
    ``mean: [0.485, 0.456, 0.406]`` rather than three. Measured, that is 52 lines
    where the block form is 108 — and it is a rule about shape, so no list of
    key names has to be kept in step with the config.
    """
    rendered = yaml.safe_dump(dict(resolved), default_flow_style=None, sort_keys=False, allow_unicode=True)
    rprint(
        Panel(
            Syntax(rendered, "yaml", theme="perldoc", background_color="default"), title="Configuration", expand=False
        )
    )


if __name__ == "__main__":
    main()
