"""What is smp-specific about a weight file: which keys are the head, and how it transplants."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from torch import nn

from src.models.heads import ExpandedHead

if TYPE_CHECKING:
    from torch import Tensor

SMP_HEAD_PREFIXES: tuple[str, ...] = ("segmentation_head.", "classification_head.")
"""The head attributes every smp model carries; keys under them are stashed, never loaded.

Constants rather than discovery, because smp names both heads by stable attributes. Only
the segmentation head transplants — the auxiliary classification head has no consumer
here, and stashing it merely keeps its keys away from the backbone.
"""

_SEGMENTATION = "segmentation_head."


def transplanted_segmentation_head(template: nn.Module, stash: dict[str, Tensor], out_features: int) -> nn.Module:
    """The carried segmentation head at the task's size: whole, or ``ExpandedHead`` when grown.

    Channels transplant by index — the declared ``classes`` pin old classes
    to their old indices. The carried head is rebuilt from the adapter's
    template (same structure, projection resized to the carried count), so the
    checkpoint loads strict; a grown task gets a second fresh template copy
    for the novel channels, concatenated on the class axis.
    """
    carried_state = {
        key.removeprefix(_SEGMENTATION): value for key, value in stash.items() if key.startswith(_SEGMENTATION)
    }
    if not carried_state:
        raise ValueError(f"Checkpoint head carries no segmentation_head tensors (found: {', '.join(sorted(stash))}).")
    path, _, _, layer = _last_projection(template)
    projection_weight = carried_state.get(f"{path}.weight")
    if projection_weight is None:
        raise ValueError(f"Checkpoint segmentation_head has no '{path}.weight'; not the template's structure.")
    carried = int(projection_weight.shape[0])
    template_in = int(layer.in_channels if isinstance(layer, nn.Conv2d) else layer.in_features)
    if int(projection_weight.shape[1]) != template_in:
        raise ValueError(
            f"Checkpoint segmentation head expects {int(projection_weight.shape[1])} features, this "
            f"decoder produces {template_in}: a foreign feature space cannot be transplanted."
        )
    if carried > out_features:
        raise ValueError(
            f"Checkpoint carries {carried} classes, the task declares {out_features}; "
            "narrowing the class space needs an explicit mapping this transplant does not guess."
        )
    base = _template_at(template, carried)
    base.load_state_dict(carried_state, strict=True)
    if carried == out_features:
        return base
    return ExpandedHead(base=base, novel=_template_at(template, out_features - carried))


def replace_last_projection(module: nn.Module, out_features: int) -> None:
    """Swap the last ``Conv2d``/``Linear`` of ``module`` for one with ``out_features`` outputs."""
    _, parent, attribute, layer = _last_projection(module)
    setattr(parent, attribute, _resized_projection(layer, out_features))


def _template_at(template: nn.Module, out_features: int) -> nn.Module:
    """A fresh copy of the head template, its projection sized to ``out_features``."""
    head = copy.deepcopy(template)
    replace_last_projection(head, out_features)
    return head


def _last_projection(module: nn.Module) -> tuple[str, nn.Module, str, nn.Conv2d | nn.Linear]:
    """The deepest-last projection layer: its dotted state path, parent, attribute, and itself."""
    found: list[tuple[str, nn.Module, str, nn.Conv2d | nn.Linear]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            path = f"{prefix}{child_name}"
            if isinstance(child, nn.Conv2d | nn.Linear):
                found.append((path, parent, child_name, child))
            else:
                visit(child, f"{path}.")

    visit(module, "")
    if not found:
        raise ValueError(f"No Conv2d or Linear found in {type(module).__name__}.")
    return found[-1]


def _resized_projection(layer: nn.Conv2d | nn.Linear, out_features: int) -> nn.Module:
    """A new layer identical to ``layer`` except for its output dimension."""
    if isinstance(layer, nn.Conv2d):
        return nn.Conv2d(
            layer.in_channels,
            out_features,
            kernel_size=layer.kernel_size,  # type: ignore[arg-type]
            stride=layer.stride,  # type: ignore[arg-type]
            padding=layer.padding,  # type: ignore[arg-type]
            bias=layer.bias is not None,
        )
    return nn.Linear(layer.in_features, out_features, bias=layer.bias is not None)


__all__ = ["SMP_HEAD_PREFIXES", "replace_last_projection", "transplanted_segmentation_head"]
