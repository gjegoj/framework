"""Putting a set of weights into a model: whole, or refused by name."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn


def load_weights(model: nn.Module, weights: Mapping[str, Tensor], source: str) -> None:
    """Put these weights into the model exactly, or refuse naming where they came from.

    Exactly, because a partial load is the failure that costs a whole run: a model with a fresh head
    and a loaded encoder looks trained and is not. torch's own missing-and-unexpected keys stand above
    the message, which is where the mismatch is actually readable.
    """
    try:
        model.load_state_dict(dict(weights))
    except RuntimeError as error:
        raise ValueError(
            f"{source} does not fit {type(model).__name__} — the missing and unexpected keys are above. "
            "Usually the run that wrote it declared a different backbone or different tasks; weights of a "
            "backbone architecture itself belong in the model section instead."
        ) from error
