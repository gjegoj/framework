"""Reading a run's checkpoint back into a model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.models import DistilledModel
from src.training import TrainingModule

if TYPE_CHECKING:
    from torch import Tensor, nn

log = logging.getLogger(__name__)


def shipped_weights(path: str) -> dict[str, Tensor]:
    """The weights a run's checkpoint holds for the model that ships.

    A run writes its whole training module, so every key is the model's own under the
    attribute it sits at — and a distilled run nests the student one level further,
    because a decorator renames what it wraps. Measured, that rename cannot be hidden
    inside the decorator: overriding ``state_dict`` propagates through the parent while
    overriding ``load_state_dict`` does not, so the two halves would disagree and a save
    would not load back.

    Unwrapping it here gives one rule for every file this framework writes: a checkpoint
    carries the weights of the model that ships, and the scaffolding a run wore —
    teachers, a criterion's own state — is not part of them. One file then loads into a
    distilled model and a plain one alike, which is what makes a later export-only run
    possible without re-declaring teachers it will not use.

    ``weights_only=True`` is enough for a Lightning checkpoint, weights-only or
    full (measured), so reading one opens no arbitrary-code surface. A file
    without a ``state_dict`` is refused by name: a backbone's own arrived weights
    are a different kind of file and belong in ``model.checkpoint_path``, where
    the adapter knows how to graft them.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError(
            f"{path} is not a checkpoint this framework wrote: it carries no 'state_dict'. "
            "Arrived weights of a backbone architecture go in 'model.checkpoint_path' instead."
        )
    model_weights = _under(state["state_dict"], TrainingModule.MODEL)
    return _under(model_weights, DistilledModel.STUDENT) or model_weights


def _under(weights: dict[str, Tensor], owner: str) -> dict[str, Tensor]:
    """The entries an attribute contributed, under the names that attribute knows them by."""
    prefix = f"{owner}."
    return {name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)}


def load_weights(model: nn.Module, path: str) -> None:
    """Put a checkpoint's weights into the model, and nothing else of it.

    Takes the model rather than the training module: what a checkpoint is *about*
    is the model, and the optimizer and the epoch counter deliberately start
    fresh — a resumed run goes through ``run.resume_path`` and Lightning instead.
    """
    try:
        model.load_state_dict(shipped_weights(path))
    except RuntimeError as error:
        raise ValueError(
            f"{path} does not fit {type(model).__name__} — the missing and unexpected keys are above. "
            "Two causes are usual: a run with 'adapters' renames every targeted layer "
            "('...proj.weight' becomes '...proj.base_layer.weight') and adds the deltas beside it, so one "
            "side of that boundary does not load into the other; and a teacher whose 'model' section is "
            "not the architecture its checkpoint was written from."
        ) from error
    log.info("Loaded the weights from %s; the optimizer and the epoch counter start fresh.", path)
