"""Checkpoint resolution and weight loading for test/export orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch

from src.config.schema import ExperimentConfig
from src.training.modules import LitModule

log = logging.getLogger(__name__)


def extract_model_state_dict(checkpoint: object) -> dict[str, Any]:
    """Pull a ``state_dict`` mapping from a Lightning ``.ckpt`` or a raw weights file.

    Parameters:
        checkpoint (object): Object returned by ``torch.load``.

    Returns:
        dict[str, Any]: Parameter/buffer tensors keyed as in ``LitModule.state_dict()``.

    Raises:
        ValueError: If the file does not look like a supported checkpoint format.
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        if isinstance(state, dict):
            return state
    if isinstance(checkpoint, dict) and checkpoint and all(isinstance(key, str) for key in checkpoint):
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint must be a Lightning .ckpt with 'state_dict' or a raw state_dict mapping.")


def load_init_weights(lit_module: LitModule, ckpt_path: str) -> None:
    """Load pretrained weights into ``lit_module`` before ``fit`` (not resume).

    Only model parameters/buffers are copied; optimizer and trainer state are untouched.
    For Lightning checkpoints saved with EMA, ``state_dict`` holds the EMA weights
    (the same tensors used for validation and checkpoint export).

    Parameters:
        lit_module (LitModule): Module to initialize.
        ckpt_path (str): Path to a ``.ckpt`` file.

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
        ValueError: If the file format is unsupported or keys do not match (``strict`` load).
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"init_ckpt_path not found: {ckpt_path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    lit_module.load_state_dict(state_dict, strict=True)
    log.info("Loaded init weights from %s (%d tensors).", ckpt_path, len(state_dict))


def resolve_test_ckpt_path(trainer: L.Trainer, config: ExperimentConfig, *, trained: bool) -> str | None:
    """Choose which checkpoint ``trainer.test`` should load.

    Priority:
        1. Explicit ``config.ckpt_path`` (required for eval-only runs).
        2. After ``fit``, ``"best"`` when ``ModelCheckpoint`` recorded a best path.
        3. ``None`` — evaluate the in-memory module (post-fit weights).

    Parameters:
        trainer (L.Trainer): Trainer that ran (or will run) ``fit``.
        config (ExperimentConfig): Validated experiment config.
        trained (bool): Whether ``trainer.fit`` completed in this run.

    Returns:
        str | None: ``ckpt_path`` argument for ``trainer.test``, or ``None``.
    """
    if config.ckpt_path is not None:
        return config.ckpt_path
    if not trained:
        return None
    callback = trainer.checkpoint_callback
    best_path = getattr(callback, "best_model_path", None) if callback is not None else None
    if best_path:
        return "best"
    return None


def resolve_ckpt_file(trainer: L.Trainer, ckpt_path: str) -> str:
    """Turn Lightning checkpoint aliases into concrete filesystem paths.

    Parameters:
        trainer (L.Trainer): Trainer with checkpoint callback state.
        ckpt_path (str): Explicit path or alias ``best`` / ``last``.

    Returns:
        str: Resolved filesystem path.

    Raises:
        ValueError: If an alias cannot be resolved.
    """
    if ckpt_path == "best":
        callback = trainer.checkpoint_callback
        path = getattr(callback, "best_model_path", None) if callback is not None else None
        if not path:
            raise ValueError("ckpt_path='best' but no best checkpoint was saved.")
        return str(path)
    if ckpt_path == "last":
        callback = trainer.checkpoint_callback
        path = getattr(callback, "last_model_path", None) if callback is not None else None
        if not path:
            raise ValueError("ckpt_path='last' but no last checkpoint was saved.")
        return str(path)
    return ckpt_path
