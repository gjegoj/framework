"""Putting weights into a model — beneath its adapters when it wears some."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.adapters import graft_base_weights

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor, nn


def load_weights(model: nn.Module, weights: Mapping[str, Tensor], source: str) -> None:
    """Put the weights into the model exactly, or refuse by name.

    A plain checkpoint meeting a model that wears adapters is the warm-start-then-LoRA
    workflow, not a mistake: the weights are grafted beneath the adapters (see
    ``graft_base_weights``). Anything else that does not fit is refused naming the source
    and the model, with torch's own missing-and-unexpected keys above the message.
    """
    try:
        model.load_state_dict(dict(weights))
    except RuntimeError as error:
        if not graft_base_weights(model, weights):
            raise ValueError(
                f"{source} does not fit {type(model).__name__} — the missing and unexpected keys are above. "
                "Three causes are usual: an adapted run's checkpoint carries '...base_layer...' and 'lora_' "
                "keys that only load back into a model declaring the same 'adapters'; a teacher whose "
                "'model' section is not the architecture its checkpoint was written from; and a checkpoint "
                "written while native heads were wrapped, whose 'heads.<task>._module.*' keys the model now "
                "holds as 'heads.<task>.*'."
            ) from error
