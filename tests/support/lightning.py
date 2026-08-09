"""A trainer that prints nothing and writes nothing — the half of a test that is not its subject.

Every knob below is one no test asserts on, and each was previously repeated, or
forgotten, per file: a run that keeps its progress bar floods the output, and one that
keeps ``enable_checkpointing`` without a root writes ``lightning_logs/`` into the
repository, which is a real thing that happened.
"""

from __future__ import annotations

from typing import Any

import lightning as L

QUIET: dict[str, Any] = {
    "max_epochs": 1,
    "accelerator": "cpu",
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}
"""What a test wants from a trainer when the trainer is not what it is testing."""


def quiet_trainer(**overrides: Any) -> L.Trainer:
    """A trainer with everything a test does not assert on turned off.

    Parameters:
        **overrides (Any): Whatever the test *is* about — more epochs, its callbacks,
            a real logger, a root to write under.

    Raises:
        ValueError: If checkpointing is turned on without a ``default_root_dir``.
            Lightning would fall back to the working directory, which under pytest is
            the repository root.
    """
    settings = QUIET | overrides
    if settings.get("enable_checkpointing") and "default_root_dir" not in settings:
        raise ValueError(
            "A checkpointing trainer needs a 'default_root_dir', or Lightning writes into "
            "the working directory — the repository root. Pass tmp_path."
        )
    return L.Trainer(**settings)
