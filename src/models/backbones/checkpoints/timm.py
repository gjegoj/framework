"""What is timm-specific about a weight file: which keys are the classifier, and how it transplants."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from src.models.heads import ExpandedHead

if TYPE_CHECKING:
    from torch import Tensor

log = logging.getLogger(__name__)


def classifier_prefixes(model: nn.Module) -> tuple[str, ...]:
    """The classifier key prefixes this architecture names for itself."""
    pretrained_cfg = cast("dict[str, Any]", getattr(model, "pretrained_cfg", {}))
    named = pretrained_cfg.get("classifier", "fc")
    names = named if isinstance(named, tuple | list) else (named,)
    return tuple(f"{name}." for name in names)


def transplanted_classifier(stash: dict[str, Tensor], in_features: int, out_features: int) -> nn.Module:
    """The carried classifier at the task's size: whole, or ``ExpandedHead`` when grown.

    Rows transplant by index — the declared ``classes`` pin old classes to
    their old indices. Narrowing, a foreign feature space, and classifier
    shapes that are not one weight/bias pair are refused by name.
    """
    weight, bias = _classifier_tensors(stash)
    carried, checkpoint_features = (int(size) for size in weight.shape)
    if checkpoint_features != in_features:
        raise ValueError(
            f"Checkpoint classifier expects {checkpoint_features} features, this head reads "
            f"{in_features}: a foreign feature space cannot be transplanted."
        )
    if carried > out_features:
        raise ValueError(
            f"Checkpoint carries {carried} classes, the task declares {out_features}; "
            "narrowing the class space needs an explicit mapping this transplant does not guess."
        )
    base = _classifier(in_features, carried)
    _fill(cast("nn.Linear", base), weight, bias)
    if carried == out_features:
        log.info("Warm-started all %d classifier rows from the checkpoint.", carried)
        return base
    log.info("Warm-started %d of %d classifier rows; %d fresh.", carried, out_features, out_features - carried)
    return ExpandedHead(base=base, novel=_classifier(in_features, out_features - carried))


def _classifier(in_features: int, out_features: int) -> nn.Module:
    """One classifier from timm's own factory — the biased Linear a checkpoint carries."""
    from timm.layers import create_classifier

    _, classifier = create_classifier(in_features, out_features)
    return cast("nn.Module", classifier)


def _classifier_tensors(stash: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """The single weight/bias pair a timm classifier carries; loud on any other shape."""
    weights = [key for key in stash if key.endswith(".weight")]
    biases = [key for key in stash if key.endswith(".bias")]
    if len(weights) != 1 or len(biases) != 1:
        raise ValueError(
            f"Checkpoint classifier is not one weight/bias pair (found: {', '.join(sorted(stash))}); "
            "not a timm classifier shape this transplant knows."
        )
    return stash[weights[0]], stash[biases[0]]


def _fill(linear: nn.Linear, weight: Tensor, bias: Tensor) -> None:
    """Transplant checkpoint tensors into a factory-built classifier.

    ``no_grad`` here is what an in-place write into a live parameter demands
    (torch refuses it otherwise) — it neither freezes nor detaches anything.
    The transplanted rows keep ``requires_grad`` and train normally until the
    freeze callback says otherwise; the same idiom sits inside ``nn.init``.
    """
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)


__all__ = ["classifier_prefixes", "transplanted_classifier"]
