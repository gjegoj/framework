"""Reading back a checkpoint this framework wrote: one run's model, out of a whole run's state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.models import load_weights

if TYPE_CHECKING:
    import lightning as L
    from torch import Tensor, nn

log = logging.getLogger(__name__)

MODEL_PREFIX = "learner.model."
"""Where the model's own entries sit inside the state a training module is checkpointed from.

Both attributes are contracts of their own — the learner under ``learner``, its network under
``model`` — and this is the one place that composes them into the prefix a saved file carries.
"""


def restore_best_weights(trainer: L.Trainer, model: nn.Module) -> None:
    """Put the checkpoint the run kept back into the model.

    Lightning does not: measured on 2.6.5, a module passed explicitly is never reloaded, so a run that
    monitored a metric would report the last epoch's numbers while keeping a different epoch on disk —
    and then ship that last epoch. A run that kept nothing has no such path, and this does nothing.
    """
    kept = getattr(trainer.checkpoint_callback, "best_model_path", "")
    if kept:
        load_checkpoint(model, str(kept))


def load_checkpoint(model: nn.Module, path: str) -> None:
    """Put a checkpoint's weights into the model, and nothing else of it.

    The model rather than the whole module: what a checkpoint is *about* is the network, and the
    optimizer and the epoch counter deliberately start fresh — continuing an interrupted run is what
    ``run.resume_path`` and Lightning are for.
    """
    load_weights(model, shipped_weights(path), path)
    log.info("Loaded the weights from %s; the optimizer and the epoch counter start fresh.", path)


def shipped_weights(path: str) -> dict[str, Tensor]:
    """The model's own weights out of a checkpoint this framework wrote.

    A run writes its whole training module, so the model's entries carry the path to it; unwrapping
    here means one file loads into any run that declares the same network. ``weights_only=True`` is
    enough for a Lightning checkpoint (measured). A file that is not one of ours is refused by name.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError(
            f"{path} is not a checkpoint this framework wrote: it carries no 'state_dict'. Weights of a "
            "backbone architecture itself belong in the model section instead."
        )
    return {
        name.removeprefix(MODEL_PREFIX): value
        for name, value in state["state_dict"].items()
        if name.startswith(MODEL_PREFIX)
    }
