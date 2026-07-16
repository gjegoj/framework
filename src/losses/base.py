"""Shared base for single-term criteria — the extension point for wrapping any loss module.

The layout convention for this package: one module per loss *family* —
``classification`` / ``regression`` / ``segmentation`` / ``composite`` and the
metric-learning families ``angular`` / ``contrastive`` / ``ranking``. A new loss goes
into its family module (or a new family module) and registers in ``criteria``.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.ports import Criterion


def require_view_shape(logits: Tensor, views: int, owner: str) -> None:
    """Validate the multi-view/multi-stream carrier shape ``[B, views, D]``.

    Shared by the ranking and contrastive families so shape mismatches surface with one
    consistent, early error instead of failing inside PyTorch's loss functions.

    Parameters:
        logits (Tensor): The candidate ``[B, N, D]`` carrier.
        views (int): Expected ``N`` (2 for pairs, 3 for triplets).
        owner (str): Criterion class name for the error message.

    Raises:
        ValueError: If ``logits`` is not ``[B, views, D]``.
    """
    if logits.ndim != 3 or logits.size(1) != views:
        raise ValueError(f"{owner} expects logits of shape [B, {views}, D], got {tuple(logits.shape)}.")


class SingleTermCriterion(Criterion):
    """Base for criteria that wrap one loss module and log it under one component.

    Subclasses build their ``nn.Module`` loss in ``__init__`` and hand it to
    ``super().__init__``; they set ``component_name`` — the label the scalar is
    logged under (conventionally the registry key).

    Wrapper convention: declare explicitly only the parameters that need conversion
    (e.g. a ``weight`` list → tensor) or a framework default; forward everything else
    verbatim to the wrapped loss via ``**kwargs`` so every upstream knob stays
    reachable from YAML without wrapper changes.

    Parameters:
        loss (nn.Module): Wrapped loss, called as ``loss(logits, target)``.
    """

    component_name: str

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self._loss = loss

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={self.component_name: value})
