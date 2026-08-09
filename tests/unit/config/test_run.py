"""``RunConfig``: what a run does and where it writes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import RunConfig


def test_a_run_trains_and_tests_by_default() -> None:
    run = RunConfig()

    assert run.train is True
    assert run.test is True
    assert run.checkpoint_path is None
    assert run.resume_path is None
    assert run.directory is None


def test_evaluating_a_checkpoint_is_expressible() -> None:
    """Judging a file is loading it and not training: no third field is needed to say so."""
    run = RunConfig.model_validate({"train": False, "checkpoint_path": "runs/best.ckpt"})

    assert run.train is False
    assert run.checkpoint_path == "runs/best.ckpt"


def test_the_two_checkpoints_differ_by_what_the_file_carries() -> None:
    """Weights alone against the whole training state — the names are the whole distinction."""
    run = RunConfig.model_validate({"resume_path": "runs/last.ckpt"})

    assert run.resume_path == "runs/last.ckpt"
    assert run.checkpoint_path is None


def test_naming_both_checkpoints_is_refused() -> None:
    """A resumed run carries its own weights, so the other file would be loaded and then overwritten."""
    with pytest.raises(ValidationError, match="not both"):
        RunConfig.model_validate({"checkpoint_path": "a.ckpt", "resume_path": "b.ckpt"})


def test_resuming_without_training_is_refused() -> None:
    """There is nothing to continue, and the user meant the other field."""
    with pytest.raises(ValidationError, match="checkpoint_path"):
        RunConfig.model_validate({"train": False, "resume_path": "runs/last.ckpt"})


def test_a_typo_in_the_run_section_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tst"):
        RunConfig.model_validate({"tst": True})


def test_run_identity_names_the_project_and_the_run() -> None:
    """One declared identity feeds the tracker, the runs/ tree, and the checkpoints root."""
    run = RunConfig.model_validate({"project": "pets", "name": "2026-08-05/10-00-00"})

    assert run.project == "pets"
    assert run.name == "2026-08-05/10-00-00"


def test_identity_is_optional() -> None:
    """Programmatic callers and tests live without a tracker; None means backend defaults."""
    run = RunConfig()

    assert run.project is None
    assert run.name is None
