"""Family-agnostic mechanics of arrived weights: load, stash, report."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

log = logging.getLogger(__name__)


def load_arrived_weights(
    model: nn.Module,
    model_name: str,
    path: str | Path,
    *,
    use_ema: bool,
    strict: bool,
    weights_only: bool,
    stash_prefixes: tuple[str, ...],
) -> dict[str, Tensor] | None:
    """Load a full checkpoint into a headless model; return the stashed head tensors.

    Every checkpoint-format concern — EMA-branch priority, ``state_dict``/``model``
    unwrapping, DDP and compile prefix stripping, safetensors — is delegated whole to
    ``timm.models.load_checkpoint``, which works on any plain ``nn.Module``. What is
    added on top: the load is reported, and a checkpoint matching nothing is refused,
    because a user's weight file must never no-op in silence.

    The knobs carry ``timm.load_checkpoint``'s own names. The stash rides its
    ``filter_fn`` hook — keys under ``stash_prefixes`` never reach the model, and
    return here for the family's own transplant to consume.

    Parameters:
        model (nn.Module): The headless model to fill.
        model_name (str): Named in the refusal, so a mismatch says which two things
            did not meet.
        path (str | Path): The weight file.
        use_ema (bool): Prefer the checkpoint's EMA branch where it has one.
        strict (bool): Refuse a state dict that does not match key for key.
        weights_only (bool): Torch's unpickling guard.
        stash_prefixes (tuple[str, ...]): Key prefixes that belong to the head.

    Returns:
        dict[str, Tensor] | None: The head tensors held back, or ``None`` if the
            checkpoint carried none.
    """
    from timm.models import load_checkpoint

    stash: dict[str, Tensor] | None = None
    offered = 0

    def stashing_head(state_dict: dict[str, Tensor], _: nn.Module) -> dict[str, Tensor]:
        nonlocal stash, offered
        stash = {key: value for key, value in state_dict.items() if key.startswith(stash_prefixes)} or None
        remaining = {key: value for key, value in state_dict.items() if not key.startswith(stash_prefixes)}
        offered = len(remaining)
        return remaining

    # torch's _IncompatibleKeys is a NamedTuple; unpacking gives the two lists
    # their own typed names (load_checkpoint itself is annotated `-> Any`).
    missing_keys, unexpected_keys = cast(
        "tuple[list[str], list[str]]",
        load_checkpoint(
            model,
            str(path),
            use_ema=use_ema,
            strict=strict,
            weights_only=weights_only,
            filter_fn=stashing_head,
        ),
    )
    loaded = offered - len(unexpected_keys)
    if loaded == 0:
        raise ValueError(f"Checkpoint '{path}' shares no weights with '{model_name}'.")
    log.info("Loaded %d tensors from '%s' (%d head tensors stashed).", loaded, path, len(stash or ()))
    if missing_keys or unexpected_keys:
        log.warning("Partial load from '%s': missing %s; unexpected %s.", path, missing_keys, unexpected_keys)
    return stash


__all__ = ["load_arrived_weights"]
