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


def restore_best_weights(trainer: L.Trainer, learner: nn.Module) -> str | None:
    """Put back what the epoch this run kept had learned, and answer with the file it was read from.

    The learner rather than the network inside it: an objective carrying parameters of its own — an
    angular margin's prototypes, a learned uncertainty — is optimized and checkpointed with the run, and
    restoring the network alone would pair one epoch's weights with another's objective, which is a
    model that existed at no point of the run.

    Lightning restores nothing here: measured on 2.6.5, a module passed explicitly is never reloaded, so
    a run would report one epoch's numbers while keeping another on disk, and then ship that one.

    The file is the answer because it is the checkpoint the run ends holding — what the export is written
    from and what a tracker is handed — and this is where that is decided. A run that kept nothing answers
    ``None``.
    """
    kept = str(getattr(trainer.checkpoint_callback, "best_model_path", ""))
    if kept:
        load_weights(learner, _under(LEARNER_PREFIX, kept), kept)
        log.info("Restored the model and the objective from %s, the epoch this run kept.", kept)
    return kept or None


def load_checkpoint(model: nn.Module, path: str) -> None:
    """Put a checkpoint's weights into the model, and nothing else of it.

    The model rather than the whole module: what a checkpoint is *about* is the network, and the
    optimizer and epoch counter deliberately start fresh — continuing an interrupted run is what
    ``run.resume_path`` and Lightning are for.

    The objective is left out for the same reason, and it is a decision rather than an omission: a run
    that trains is going to learn its own, and the file may well come from one learned under another.
    A run that only scores has no such chance, and reads the file through ``load_learned`` instead.
    """
    load_weights(model, model_weights(path), path)
    log.info("Loaded the weights from %s; the optimizer and the epoch counter start fresh.", path)


def load_learned(learner: nn.Module, path: str) -> None:
    """Put back everything a checkpoint holds of what a run learned: its network, and its objective.

    For a run that scores without training. An objective carrying parameters of its own — an angular
    margin's prototypes, a learned uncertainty — is optimized and written with the run that learned
    them, and a run that does not train has nothing to learn them with. Restoring the network alone
    leaves them at whatever ``seed`` produced, and then the number reported under that objective's name
    is a reading of the seed rather than of this file: measured on a five-epoch metric-learning run, the
    same checkpoint read back reported 33.11, 43.00 and 41.80 at seeds 42, 7 and 1234, against the 33.04
    the run itself reported. Recall@1 was 0.3333 in all four, because it is read off the network.

    What is missing is refused here rather than left to ``load_weights`` below, which guards the same
    file against different harm — a shape that does not fit, an entry nothing has a place for — and
    says so in those terms. The harm here is that the run would report at all.
    """
    held = _under(LEARNER_PREFIX, path)
    if absent := sorted(set(learner.state_dict()) - set(held)):
        raise ValueError(
            f"{path} carries nothing for {', '.join(absent)}, and this run does not train: every number "
            f"it reported about those would be a reading of `seed` rather than of this file. Score a "
            f"checkpoint written by a run declared like this one, or declare `run.train: true` so that "
            f"what the file does not carry is learned rather than reported."
        )
    load_weights(learner, held, path)
    log.info("Loaded the network and the objective from %s, which is what this run reports on.", path)


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
