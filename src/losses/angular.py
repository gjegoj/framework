"""Angular-margin losses for metric learning (ArcFace family).

Two flavors, two prototype placements. The **classifier** flavor consumes the cosine
logits ``[B, C]`` produced by a ``cosine`` head and the integer class labels, then
applies a margin in angular space before cross-entropy — the learnable class
prototypes live in the head, so the criterion itself is **stateless** (margin and
scale are fixed hyper-parameters); adding CosFace/SphereFace later is just another
stateless criterion over the same cosine head. The **embedder** flavor
(``ProxyAngularCriterion``) instead holds the prototypes itself, training-only: it
wraps a stateless margin criterion with its own learnable class prototypes so a plain
embedding head can be trained via proxy classification and the prototypes discarded
at export.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult
from src.core.instantiate import BrickSpec, instantiate
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
        self._cos_margin = math.cos(margin)
        self._sin_margin = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)  # below this, use the linear fallback
        self._linear_margin = math.sin(math.pi - margin) * margin  # penalty for the linear-fallback branch

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        if logits.ndim != 2:
            raise ValueError(f"ArcFaceCriterion expects cosine logits of shape [B, C], got {tuple(logits.shape)}.")
        cosine = logits.clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine**2).clamp_min(0.0))
        phi = cosine * self._cos_margin - sine * self._sin_margin  # cos(θ + m)
        # Keep the target term monotonic for large angles (ArcFace, easy_margin=False).
        phi = torch.where(cosine > self._threshold, phi, cosine - self._linear_margin)

        labels = target.long()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self._scale

        value: Tensor = F.cross_entropy(output, labels)
        return LossResult(total=value, components={"arcface": value})


@criteria.register("arcface_proxy")
class ProxyAngularCriterion(Criterion):
    """Training-only cosine classifier over learnable class prototypes (ArcFace-as-embedder).

    The original-paper recipe: train an embedding with a proxy classification head, then
    discard the head and deploy only the embedder. The "head" here is deliberately inside
    the criterion — it must never enter the exported graph — while the margin math is
    delegated to a stateless inner criterion (``arcface`` by default), so CosFace later is
    a YAML ``inner:`` change, not new proxy code.

    ``requires_dimensions``: the task layer injects both sizes at build time —
    ``num_classes`` from the fitted label encoder, ``embedding_dim`` from the head size.

    Parameters:
        num_classes (int): Number of class prototypes (label-vocabulary size from data).
        embedding_dim (int): Embedding size D — must match the task head's out_features.
        inner (BrickSpec): Stateless margin criterion applied to the cosine logits.
    """

    requires_dimensions = True

    def __init__(self, num_classes: int, embedding_dim: int, inner: BrickSpec = "arcface") -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.prototypes)
        self._margin_criterion: Criterion = instantiate(inner, criteria)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        # ``logits`` carries the [B, D] embedding (the head output for metric tasks).
        cosine = F.normalize(logits, dim=1) @ F.normalize(self.prototypes, dim=0)
        return self._margin_criterion(cosine, target)
