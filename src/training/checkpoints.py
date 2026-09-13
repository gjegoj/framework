"""Reading back a checkpoint this framework wrote: one run's model, out of a whole run's state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.models import load_weights
from src.training.module import TrainingModule

if TYPE_CHECKING:
    import lightning as L
    from torch import Tensor, nn

log = logging.getLogger(__name__)

MODEL_PREFIX = f"{TrainingModule.MODEL}."
"""Where the model's own entries sit inside the state a training module is checkpointed from.

The path is the module's to declare — a config's freeze path is written against the same one — and this
is where it becomes the prefix a saved file's keys carry.
"""

LEARNER_PREFIX = f"{TrainingModule.LEARNER}."
"""The same, one level up: everything the run learned, rather than the network alone."""


def restore_best_weights(trainer: L.Trainer, learner: nn.Module) -> None:
    """Put back what the epoch this run kept had learned. A run that kept nothing does nothing here.

    The learner rather than the network inside it: an objective carrying parameters of its own — an
    angular margin's prototypes, a learned uncertainty — is optimized and checkpointed with the run, and
    restoring the network alone would pair one epoch's weights with another's objective, which is a
    model that existed at no point of the run.

    Lightning restores nothing here: measured on 2.6.5, a module passed explicitly is never reloaded, so
    a run would report one epoch's numbers while keeping another on disk, and then ship that one.
    """
    kept = str(getattr(trainer.checkpoint_callback, "best_model_path", ""))
    if kept:
        load_weights(learner, _under(LEARNER_PREFIX, kept), kept)
        log.info("Restored the model and the objective from %s, the epoch this run kept.", kept)


def load_checkpoint(model: nn.Module, path: str) -> None:
    """Put a checkpoint's weights into the model, and nothing else of it.

    The model rather than the whole module: what a checkpoint is *about* is the network, and the
    optimizer and epoch counter deliberately start fresh — continuing an interrupted run is what
    ``run.resume_path`` and Lightning are for.
    """
    load_weights(model, model_weights(path), path)
    log.info("Loaded the weights from %s; the optimizer and the epoch counter start fresh.", path)


def model_weights(path: str) -> dict[str, Tensor]:
    """The model's own weights out of a checkpoint this framework wrote, its path prefix removed.

    A run writes its whole training module, so unwrapping here is what lets one file load into any run
    that declares the same network.
    """
    return _under(MODEL_PREFIX, path)


def _under(prefix: str, path: str) -> dict[str, Tensor]:
    """A checkpoint's entries below one path in the module that wrote it, that path removed.

    ``weights_only=True`` is enough for a Lightning checkpoint (measured); a file that is not one of
    ours is refused by name.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError(
            f"{path} is not a checkpoint this framework wrote: it carries no 'state_dict'. Weights of a "
            "backbone architecture itself belong in the model section instead."
        )
    return {name.removeprefix(prefix): value for name, value in state["state_dict"].items() if name.startswith(prefix)}
