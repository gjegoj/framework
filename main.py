"""Training entry point.

Usage::

    uv run main.py experiment=examples/classification
    uv run main.py experiment=examples/classification lr=3e-4 epochs=50
    uv run main.py experiment=examples/classification run.train=false +run.checkpoint_path=runs/best.ckpt
    uv run main.py experiment=examples/classification +run.resume_path=runs/last.ckpt

A key the composed config already declares is overridden by name; one it does not —
``run.checkpoint_path``, ``run.resume_path`` — is *added* with Hydra's ``+``.

The work lives in :mod:`src.cli`; this file only makes the repository root a place a
run can be started from, so ``uv run main.py`` and ``uv run python -m src.cli`` are the
same run. ``configs/`` is found by absolute path either way.
"""

from __future__ import annotations

from src.cli import main

if __name__ == "__main__":
    main()
