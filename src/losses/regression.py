"""Regression criteria."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, override

import torch
from torch import nn

from src.core.entities import Loss
from src.core.ports import Criterion
from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor


class _RegressionCriterion(WrappedCriterion):
    """Shared shape handling: a single-output head against channel-free targets.

    ``[B, 1]`` outputs vs ``[B]``, or dense ``[B, 1, H, W]`` vs ``[B, H, W]`` —
    the channel is squeezed so a silent broadcast cannot happen; matching
    shapes pass through.
    """

    @override
    def _prepare(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if logits.dim() == target.dim() + 1:
            logits = logits.squeeze(1)  # The channel dim: [B, 1] and [B, 1, H, W] alike.
        return logits, target


@criterion_registry.register("mse")
class MeanSquaredErrorCriterion(_RegressionCriterion):
    """Mean squared error on raw outputs.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.MSELoss``.
    """

    part_name: ClassVar[str] = "mse"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.MSELoss(**kwargs))


@criterion_registry.register("mae")
class MeanAbsoluteErrorCriterion(_RegressionCriterion):
    """Mean absolute error (torch's ``L1Loss``) on raw outputs.

    The outlier-tolerant sibling of ``mse``: an error counts once, not squared.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.L1Loss``.
    """

    part_name: ClassVar[str] = "mae"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.L1Loss(**kwargs))


@criterion_registry.register("huber")
class HuberCriterion(_RegressionCriterion):
    """Huber loss on raw outputs: quadratic near zero, linear past ``delta``.

    The compromise between ``mse`` and ``mae`` — precise on small errors, calm
    about outliers.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.HuberLoss`` (``delta``, ...).
    """

    part_name: ClassVar[str] = "huber"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.HuberLoss(**kwargs))


@criterion_registry.register("smooth_l1")
class SmoothL1Criterion(_RegressionCriterion):
    """Smooth L1 on raw outputs: Huber's shape, scaled inside ``beta``.

    Identical to ``huber`` at ``beta == delta == 1``; they diverge in how the
    quadratic zone is scaled — torch keeps both, and so do we, because papers
    cite them by these exact names (detection heads say Smooth L1).

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.SmoothL1Loss`` (``beta``, ...).
    """

    part_name: ClassVar[str] = "smooth_l1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.SmoothL1Loss(**kwargs))


@criterion_registry.register("expectation")
class ExpectationCriterion(Criterion):
    """Regression on the expectation of a distribution over ordered classes.

    Both sides are reduced to one number — ``softmax(logits) · class_values``
    against the same weighting of the target — and compared by an ordinary
    regression criterion. It is the companion of cross-entropy for binned
    targets, not a replacement: cross-entropy shapes the whole distribution but
    saturates once a prediction stops overlapping the target, while this term
    keeps a signal proportional to how far the reported number actually is.
    Alone it is just as insufficient — any distribution with the right mean
    satisfies it.

    ``class_values`` is never written by hand: the target encoder that laid out
    the bins reports them, and assembly passes them through as a derived value.

    Two numbers can be compared in more than one way, so ``distance`` is a
    criterion slot — absolute error by default, Huber when outliers should not
    dominate::

        loss:
          - {name: cross_entropy}
          - name: expectation
            weight: 0.5
            distance: {_target_: src.losses.HuberCriterion, delta: 0.1}

    Whichever distance compares the numbers, the term logs as ``expectation``:
    that is its identity in a composite, and the metric inside is its detail.

    Parameters:
        class_values (list[float]): The number each class position stands for.
        distance (Criterion | None): How the two numbers are compared;
            ``None`` builds the default ``mae`` from the remaining arguments.
        **kwargs: Forwarded verbatim to the default ``mae``.
    """

    part_name: ClassVar[str] = "expectation"

    def __init__(self, class_values: list[float], distance: Criterion | None = None, **kwargs: Any) -> None:
        super().__init__()
        if distance is not None and kwargs:
            raise ValueError(
                f"expectation takes either a 'distance' criterion or arguments for the default mae, "
                f"not both; declare {sorted(kwargs)} on the module itself."
            )
        self._distance = distance if distance is not None else MeanAbsoluteErrorCriterion(**kwargs)
        values = torch.as_tensor(class_values, dtype=torch.float)
        # Non-persistent: the values belong to the data, not to the trained weights,
        # so they follow device moves without entering the checkpoint.
        self.register_buffer("class_values", values, persistent=False)
        self.class_values: Tensor

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        if logits.shape[-1] != self.class_values.numel():
            raise ValueError(
                f"expectation has {self.class_values.numel()} class values but the head produced "
                f"{logits.shape[-1]} outputs; the bins and the head must agree."
            )
        prediction = logits.softmax(dim=-1) @ self.class_values
        wanted = target.float() @ self.class_values
        return Loss.part(self.part_name, self._distance(prediction, wanted).total)
