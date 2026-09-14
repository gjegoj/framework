"""Putting a set of weights into a model: whole, or refused by name."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


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


def weights_in(path: str) -> dict[str, Tensor]:
    """Every tensor a weight file holds, whatever the tool that wrote it wrapped them in.

    A file that names an architecture: what a hub hosts, what a pretraining script saved. Not
    ``model_weights``, which asks the opposite question — what did *this framework* write — and refuses
    anything else by name, so that a run's own file is never read as a bare set of weights. Two
    questions, and a file answering one is the wrong file for the other.
    """
    held = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(held, Mapping) and "state_dict" in held:
        held = held["state_dict"]
    if not isinstance(held, Mapping) or not all(isinstance(value, Tensor) for value in held.values()):
        raise ValueError(f"{path} holds no weights this framework can read: a file of named tensors was expected.")
    return dict(held)


def start_from(network: nn.Module, path: str, *, inside: str = "", aside: Sequence[str] = ()) -> dict[str, Tensor]:
    """Fill a network from a file naming an architecture, and hand back what it had no place for.

    Two facts line a file up with a network, and both belong to the family rather than to the
    declaration that names the file: where inside this module the library's own graph sits, and which
    of the file's names are a head this graph does not carry. Measured — a resnet18 file shares none of
    its 120 names with ``TimmBackbone`` and all 120 once ``model.`` is accounted for, while an smp file
    shares 180 of 180 with ``SmpBackbone`` and brings only its segmentation head on top.

    Whole or refused, which is the rule ``load_weights`` keeps and for the same reason. What torch's own
    leniency buys instead is measured too: a file of another architecture moves **0 of 120** tensors,
    reports success, and the run then trains a random encoder with every number looking ordinary.

    Parameters:
        network: The module to fill, which is the backbone itself.
        path: The weight file, as the declaration wrote it.
        inside: Where under this module the library's graph sits, as a key prefix.
        aside: Key prefixes the file may carry that this network has no place for.
    """
    held = weights_in(path)
    carried = {name: value for name, value in held.items() if aside and name.startswith(tuple(aside))}
    offered = {f"{inside}{name}": value for name, value in held.items() if name not in carried}
    wanted = set(network.state_dict())
    missing, unexpected = sorted(wanted - set(offered)), sorted(set(offered) - wanted)
    if not wanted - set(missing):
        raise ValueError(
            f"`checkpoint_path` {path!r} shares no weights with {type(network).__name__}: it names "
            f"{', '.join(sorted(held)[:3])}… and this network is built of {', '.join(sorted(wanted)[:3])}… "
            f"A file of another architecture loads nothing at all, and nothing would say so."
        )
    if missing or unexpected:
        raise ValueError(
            f"`checkpoint_path` {path!r} fits {type(network).__name__} only in part — it carries nothing "
            f"for {', '.join(missing) or 'everything here'}, and this network has no place for "
            f"{', '.join(unexpected) or 'nothing else'}. A network half from a file is not that file's "
            f"network; declare the variant it was written from, or name the head it carries as one to "
            f"hold back."
        )
    load_weights(network, offered, path)
    log.info(
        "%s starts from the %d weights in %s; %d were held back.",
        type(network).__name__,
        len(offered),
        path,
        len(carried),
    )
    return carried
