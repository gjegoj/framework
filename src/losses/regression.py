"""Regression criteria: MSE/L1 on raw outputs, and expectation regression over bins."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.base import SingleTermCriterion
from src.losses.registry import criteria


@criteria.register("mse")
class MSECriterion(SingleTermCriterion):
    """Mean squared error on raw outputs vs float targets.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.MSELoss`` (``reduction``, ...).
    """

    component_name = "mse"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.MSELoss(**kwargs))


@criteria.register("l1")
class L1Criterion(SingleTermCriterion):
    """Mean absolute error on raw outputs vs float targets.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.L1Loss`` (``reduction``, ...).
    """

    component_name = "l1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.L1Loss(**kwargs))


_DISTRIBUTION_MEAN_KINDS: dict[str, type[nn.Module]] = {"l1": nn.L1Loss, "huber": nn.HuberLoss}


@criteria.register("distribution_mean")
class DistributionMeanCriterion(Criterion):
    """Regression on the mean of a binned label distribution (LDL / DFL-style).

    Both prediction and target are distributions over the same bins: the prediction's
    expectation ``softmax(logits) · bin_centers`` is regressed onto the target
    distribution's expectation with an L1 or Huber loss. Combine with soft-label
    cross-entropy via ``weighted_sum`` (both terms share the same logits/target) —
    the composite loss for distributional regression on one head.

    Parameters:
        bin_centers (list[float] | None): Explicit bin centers (one per class).
        bin_edges (list[float] | None): Bin boundaries; centers become the midpoints —
            paste the same list as the ``gaussian_bins`` encoder's. Exactly one of
            ``bin_centers``/``bin_edges`` is required.
        kind (str): Inner loss — ``l1`` (default) or ``huber``.
        **kwargs: Forwarded verbatim to the inner torch loss (e.g. Huber ``delta``).
    """

    def __init__(
        self,
        bin_centers: list[float] | None = None,
        bin_edges: list[float] | None = None,
        kind: str = "l1",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if (bin_centers is None) == (bin_edges is None):
            raise ValueError("distribution_mean needs exactly one of 'bin_centers' or 'bin_edges'.")
        if kind not in _DISTRIBUTION_MEAN_KINDS:
            raise ValueError(f"distribution_mean kind must be one of {sorted(_DISTRIBUTION_MEAN_KINDS)}, got {kind!r}.")
        if bin_centers is not None:
            centers = torch.as_tensor(bin_centers, dtype=torch.float)
        else:
            edges = torch.as_tensor(bin_edges, dtype=torch.float)
            centers = (edges[:-1] + edges[1:]) / 2.0
        # Non-persistent buffer: follows device moves without entering the state_dict.
        self.register_buffer("bin_centers", centers, persistent=False)
        self.bin_centers: Tensor
        self._loss = _DISTRIBUTION_MEAN_KINDS[kind](**kwargs)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.shape[-1] != self.bin_centers.numel():
            raise ValueError(
                f"distribution_mean has {self.bin_centers.numel()} bin centers but got logits with "
                f"{logits.shape[-1]} classes — the bins must match the head."
            )
        predicted_mean = (logits.softmax(dim=-1) * self.bin_centers).sum(dim=-1)
        target_mean = (target * self.bin_centers).sum(dim=-1)
        value: Tensor = self._loss(predicted_mean, target_mean)
        return LossResult(total=value, components={"distribution_mean": value})
