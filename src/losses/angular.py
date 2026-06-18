"""Angular-margin losses for metric learning (ArcFace family).

These consume the cosine logits ``[B, C]`` produced by a ``cosine`` head and the
integer class labels, then apply a margin in angular space before cross-entropy.
The learnable class prototypes live in the head; the loss itself is **stateless**
(margin and scale are fixed hyper-parameters), so adding CosFace/SphereFace later
is just another stateless criterion over the same cosine head.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.registry import criteria


@criteria.register("arcface")
class ArcFaceCriterion(Criterion):
    """ArcFace additive angular margin loss on cosine logits ``[B, C]``.

    Adds an angular margin ``m`` to the target class — ``cos(θ_y + m)`` — then
    scales by ``s`` and applies cross-entropy.  Penalizing the target in angle
    space forces tighter intra-class / wider inter-class angular separation than
    plain softmax.  The non-target logits are left untouched; the standard
    monotonicity guard (``easy_margin=False``) keeps the target term well-behaved
    for angles past ``π - m``.

    Margin and scale are fixed (no learnable state) — the only trained parameters
    are the class prototypes in the ``cosine`` head.

    Parameters:
        margin (float): Additive angular margin in radians (ArcFace default 0.5).
        scale (float): Logit scale ``s`` (ArcFace default 64).

    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition" (2019).
    """

    def __init__(self, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self._margin = margin
        self._scale = scale
        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)  # below this, use the linear fallback
        self._mm = math.sin(math.pi - margin) * margin

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 2:
            raise ValueError(f"ArcFaceCriterion expects cosine logits of shape [B, C], got {tuple(logits.shape)}.")
        cosine = logits.clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine**2).clamp_min(0.0))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(θ + m)
        # Keep the target term monotonic for large angles (ArcFace, easy_margin=False).
        phi = torch.where(cosine > self._threshold, phi, cosine - self._mm)

        labels = target.long()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self._scale

        value: Tensor = F.cross_entropy(output, labels)
        return LossResult(total=value, components={"arcface": value})
