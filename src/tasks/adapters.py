"""Built-in target adapters: shaping one raw target into its loss and metric views."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core.entities import AdaptedTarget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.ports import TargetAdapter

CLASS_DIM = 1
"""Where the classes sit in a soft target — a mix is per sample, never per pixel."""


def as_class_indices(target: Tensor) -> AdaptedTarget:
    """Class indices for both views — unless the target is already a distribution.

    A multiclass target arrives as an index (integral by construction) or, after a batch
    transform mixed two samples, as a share of each class — told apart by dtype, not a
    flag. Cross-entropy takes a distribution as it is; metrics rank against the class the
    sample mostly is. Augmentation hands masks back as ``int32`` while criteria want ``long``.
    """
    if target.is_floating_point():
        return AdaptedTarget(for_loss=target, for_metrics=target.argmax(dim=CLASS_DIM))
    indices = target.long()
    return AdaptedTarget(for_loss=indices, for_metrics=indices)


def float_for_loss(target: Tensor) -> AdaptedTarget:
    """Float the loss view (BCE needs floats); metrics keep the raw dtype.

    For a target with no hard form to round to — a price, a temperature — which
    is why an indicator uses ``as_indicators`` instead.
    """
    return AdaptedTarget(for_loss=target.float(), for_metrics=target)


def as_indicators(target: Tensor) -> AdaptedTarget:
    """Float for the loss, a 0/1 label for the metrics.

    A binary or multilabel target says whether each label applies, and a batch
    transform turns that into the share of it the mixed sample carries. Binary
    cross-entropy takes the share directly — it is what the sample is — while
    metrics rank against a label, so the metric view is the label the sample
    mostly carries. A target that never went through a mix rounds to itself.
    """
    return AdaptedTarget(for_loss=target.float(), for_metrics=(target >= 0.5).long())


def expectation_of(class_values: Sequence[float]) -> TargetAdapter:
    """Build an adapter for a target encoded as a distribution over ordered classes.

    The two views genuinely differ here: the loss compares distributions, while
    a metric compares the numbers they stand for. Both come from the same
    weighting, so a target's metric view is the value it was encoded from.

    Parameters:
        class_values (Sequence[float]): The number each class position stands for.
    """
    values = torch.as_tensor(list(class_values), dtype=torch.float)

    def adapt(target: Tensor) -> AdaptedTarget:
        distribution = target.float()
        return AdaptedTarget(for_loss=distribution, for_metrics=distribution @ values.to(target.device))

    return adapt
