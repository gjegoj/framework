"""What the entry point shows before it hands a config on, and what it lets through."""

from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.warnings import PossibleUserWarning
from omegaconf import OmegaConf
from pydantic import ValidationError
from rich.console import Console

from src.cli import CONFIG_DIRECTORY, main, show_config, silence_third_party_notices
from tests.support.datasets import TABLE_NAME

if TYPE_CHECKING:
    from omegaconf import DictConfig


def test_the_config_directory_is_the_one_holding_the_groups_a_run_composes() -> None:
    """An absolute path is returned verbatim by Hydra, so a wrong one fails only at run time.

    Nothing else checks it: every test below builds its mapping in memory, and the run
    that would notice is an e2e one. A path off by a single directory therefore reaches
    users as "Cannot find primary config 'config'" and nothing sooner.
    """
    groups = Path(CONFIG_DIRECTORY)

    assert (groups / "config.yaml").is_file()
    assert (groups / "experiment").is_dir()


def composed() -> DictConfig:
    """A config shaped like a real one: shared values reached through interpolation."""
    return OmegaConf.create(
        {
            "lr": 3.0e-4,
            "image_size": [224, 224],
            "optimizer": {"name": "adamw", "lr": "${lr}"},
            "transforms": {"train": {"height": "${image_size.0}"}},
        }
    )


def test_the_shown_config_is_yaml_rather_than_a_rendered_object(capsys: pytest.CaptureFixture[str]) -> None:
    """A config is YAML, so what reaches the screen has to be pasteable back into a file."""
    show_config({"model": {"name": "timm", "pretrained": False}})

    shown = capsys.readouterr().out

    assert "model:" in shown
    assert "name: timm" in shown
    assert "{'model'" not in shown


def test_the_shown_config_carries_the_values_a_run_read_not_the_references(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hydra archives the composed config with `${lr}` intact, so this is the only place the numbers appear."""
    resolved = cast("dict[str, Any]", OmegaConf.to_container(composed(), resolve=True))

    show_config(resolved)

    shown = capsys.readouterr().out
    assert "${" not in shown
    assert shown.count("0.0003") == 2
    assert "height: 224" in shown


def test_a_list_of_numbers_stays_on_its_own_line(capsys: pytest.CaptureFixture[str]) -> None:
    """A config read on screen has to be short enough to read; block style triples every such line.

    This is a rule about shape — a collection holding no collections stays inline —
    rather than a list of key names that would have to be kept in step with the config.
    """
    show_config({"mean": [0.485, 0.456, 0.406], "transforms": {"train": [{"name": "resize", "height": 224}]}})

    shown = capsys.readouterr().out

    assert "mean: [0.485, 0.456, 0.406]" in shown
    assert "height: 224" in shown  # the nested mapping still gets a line of its own


def test_a_config_that_is_refused_is_shown_anyway(capsys: pytest.CaptureFixture[str]) -> None:
    """The refused config is the one worth reading, and the refusal names a key without its surroundings."""
    with pytest.raises(ValidationError):
        main.__wrapped__(OmegaConf.create({"lr": 3.0e-4, "model": {"name": "timm"}}))

    assert "lr: 0.0003" in capsys.readouterr().out


def test_main_runs_the_composed_experiment_through_fit_and_test(dataset_root: Path) -> None:
    """Past the shown config: the shipped groups compose into a run that trains, tests and logs under the run's home.

    The overrides are the ones a user types — a source, a task, a size, a home — and nothing
    below ``main`` is stubbed, so this is the path ``uv run main.py`` takes.
    """
    home = dataset_root / "run"
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIRECTORY):
        composed = compose(
            config_name="config",
            overrides=[
                f"data.source={dataset_root / TABLE_NAME}",
                f"+data.inputs.image.loader.root={dataset_root}",
                "data.split={train:0.5,val:0.25,test:0.25}",
                "+tasks.label.kind=classification",
                "+tasks.label.target=label",
                "+tasks.label.classes={0:cat,1:dog}",
                "model.pretrained=false",
                "epochs=1",
                "batch_size=2",
                "image_size=[16,16]",
                f"run.directory={home}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "+trainer.enable_progress_bar=false",
                "+trainer.enable_model_summary=false",
            ],
        )

    main.__wrapped__(composed)

    logged = list(home.rglob("metrics.csv"))
    assert logged, "Lightning's own logger writes the run's metrics under the run's home"
    assert "test/label/ce" in logged[0].read_text()  # the loss part every classification run logs at test


def raised(*notices: tuple[str, type[Warning]]) -> list[str]:
    """The notices that survive the filter, raised inside a scope that restores it after."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_third_party_notices()
        for message, category in notices:
            warnings.warn(message, category, stacklevel=1)
    return [str(entry.message) for entry in caught]


def test_the_notice_about_one_library_calling_another_is_dropped() -> None:
    """Traced to `CombinedLoader.__init__`, so it arrives on every run whatever the data — and cannot be acted on."""
    survivors = raised(
        ("`isinstance(treespec, LeafSpec)` is deprecated, use `treespec.is_leaf()` instead.", FutureWarning),
    )

    assert survivors == []


def test_a_run_that_silently_skips_validation_still_says_so() -> None:
    """`PossibleUserWarning` mixes speed tips with this, so the category is no shortcut for the filter."""
    survivors = raised(
        ("You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.", PossibleUserWarning),
        ("Your `val_dataloader`'s sampler has shuffling enabled.", PossibleUserWarning),
    )

    assert len(survivors) == 2


def test_the_tips_about_this_runs_own_choices_still_arrive() -> None:
    """Both answer to config we wrote, so hiding them would hide a decision rather than someone else's chatter."""
    survivors = raised(
        ("The 'train_dataloader' does not have many workers which may be a bottleneck.", PossibleUserWarning),
        ("GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.", PossibleUserWarning),
    )

    assert len(survivors) == 2


def test_the_shown_config_goes_through_the_frameworks_one_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich reads the terminal once per console; a second one opened just for the banner can disagree
    about the width with everything the run prints after it."""
    drawn = StringIO()
    monkeypatch.setattr("src.cli.console", lambda: Console(file=drawn, width=80))

    show_config({"model": {"name": "timm"}})

    assert "name: timm" in drawn.getvalue()
