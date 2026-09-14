"""The command line: Hydra composes the declaration, the root wires it, the trainer runs it.

Hydra and OmegaConf are confined to this module — they compose the YAML groups and apply the
overrides, and everything below receives one validated ``ExperimentConfig``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.build import build
from src.config import load_config
from src.console import console
from src.experiment import run
from src.export import Manifest

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
    silence_third_party_notices()
    show(composed)
    # `throw_on_missing`, because the root config marks `data.source` as `???`: without it the marker
    # degrades into that literal string, and a run that forgot its source is told its file format is
    # wrong. Shown first, so the panel is there to read when the refusal comes.
    resolved = cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True, throw_on_missing=True))
    manifest = run(build(load_config(resolved)))
    if manifest.artifacts:
        console().print(table_for(manifest))


def table_for(manifest: Manifest) -> Table:
    """What the run wrote and how far each artifact stood from the model it was written from.

    Here rather than beside the export, which owns no presentation: a verdict that exists only as
    printed text cannot be asserted, and a record that can only be printed cannot be sent anywhere else.
    This is the third reader of one record — the two that are not a person being the file beside the
    artifacts and whatever the run declared as a tracker.

    There is no verdict column, because there is no verdict to show: an artifact outside its tolerance
    stops the run rather than reaching this table. What stands in its place is the share of the
    allowance each one used, which is the very number the refusal would have been made of.
    """
    built = Table(title="Shipped", show_header=True)
    built.add_column("Artifact")
    built.add_column("Travels with")
    built.add_column("Proven at", justify="right")
    built.add_column("Worst difference", justify="right")
    built.add_column("Of its allowance", justify="right")
    for one in manifest.artifacts:
        built.add_row(
            one.artifact,
            ", ".join(one.travels_with) or "nothing",
            ", ".join(str(batch) for batch in one.parity.batches),
            f"{one.parity.difference:.2e}",
            f"{one.parity.allowance_used:.1%}",
        )
    return built


def silence_third_party_notices() -> None:
    """Drop the notices about how one library calls another, and only those.

    A run's own substitutions stay: torch saying that MPS ignores the pinned memory a `loader` asked
    for is about this run's declaration, and the reader is the one who can act on it. What goes is the
    deprecation one library owes another — nobody reading it can do anything but wait for a release.
    Each line names one message rather than a category, so a filter outlives the notice it was written
    for by doing nothing at all instead of by hiding its neighbours.

    The second is Lightning telling a run that the module's ``on_after_batch_transfer`` wins over the
    data module's. It is emitted whenever the module overrides that hook at all, whether or not a data
    module offers one — read in its connector — and in this framework none ever does: ``TrainingData``
    is the only data module a run is assembled with, and it implements no hook the module could win
    over. What the notice says is being ignored is therefore the base class's own empty body. The
    condition that makes it vacuous is pinned by
    ``test_the_data_adapter_implements_no_hook_the_module_would_win_over``, so a hook added there turns
    this line from harmless into wrong, loudly, rather than quietly.

    The third is the stated exception to the rule above, and is here by the owner's decision. It *is* a
    substitution of this run's own numbers: a normalized confusion matrix divides each row by that
    class's support, and a class no row of the split carried divides by zero. What makes it droppable
    rather than hidden is that the matrix already shows it — the row those zeros fill is empty — and
    that a run reports one per stage, saying the same thing about the same split each time.
    """
    warnings.filterwarnings("ignore", message=r".*LeafSpec.*is deprecated", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r"You have overridden `on_after_batch_transfer` in `LightningModule`",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*NaN values found in confusion matrix have been replaced with zeros",
        category=UserWarning,
    )


def show(composed: DictConfig) -> None:
    """Show the run what it was given, in the language a declaration is written in.

    Dumped through PyYAML rather than ``OmegaConf.to_yaml``, for one setting: ``default_flow_style=None``
    puts a collection holding no collections on one line — ``image_size: [224, 224]``,
    ``classes: {0: cat, 1: dog}`` — and keeps the block form for everything nested. OmegaConf writes
    block style throughout, which turns three numbers into three lines and a vocabulary of 37 breeds
    into a panel nothing else fits beside.

    ``word_wrap`` is not decoration: a line a panel cannot fit is *cropped* by rich rather than folded,
    and measured at sixty columns a seven-key ``trainer:`` lost three of its entries out of the middle
    of the line. Wrapping the dump to the console's own width instead was tried and dropped — with the
    fold in place nothing is lost either way, and it was a setting no reader could tell apart.

    Resolved without ``throw_on_missing``, because this is printed *before* the strict read: a run that
    forgot a mandatory value should have its declaration on screen when it is told so, and here the
    marker shows as the ``'???'`` it is.
    """
    rendered = yaml.safe_dump(
        cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True)),
        default_flow_style=None,
        sort_keys=False,
        allow_unicode=True,
    )
    console().print(
        Panel(
            Syntax(rendered, "yaml", theme="perldoc", background_color="default", word_wrap=True),
            title="Run",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
