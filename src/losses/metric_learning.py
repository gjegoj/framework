"""Angular margins: objectives that separate identities rather than tell a declared vocabulary apart.

The margin scores the true class as though the sample sat further from it than it does, so the run has
to overshoot and leaves a gap behind. Where the prototypes live is the difference between the two names
below and is a deployment choice: ``arcface`` reads the cosines a ``cosine`` head produced, so they
travel with the network; ``arcface_proxy`` holds them itself, so the artifact answers with a direction.
"""

from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, normalize, one_hot

from src.core import FEATURE_AXIS, LossOutput
from src.losses.base import Loss
from src.losses.registry import loss_registry

COSINE_LIMIT = 1.0 - 1e-7
"""The bound a cosine is held inside before ``arccos`` is taken of it.

``arccos`` has unbounded derivative at ±1, which is where a converged sample sits, so without this the
gradient through the embedding is infinite. Float32 on purpose: in bfloat16 and float16 this value
rounds to exactly 1.0, the clamp stops clamping, and gradients come back NaN under a loss still
reading 0.0 — hence the float32 region in ``forward``. Both measured.
"""

COSINE_CEILING = 1.01
"""Above this, what arrived is a projection rather than a cosine that drifted.

Slack rather than exactly one, because a normalized product lands a float step past the bound; wide
slack, because the two cases this tells apart differ by orders of magnitude rather than by rounding.
"""


@loss_registry.register("arcface")
class ArcFace(Loss):
    """Cross-entropy over cosines, with an angular margin added to the identity a sample already is.

    Reads one cosine per class, which is what a ``cosine`` head produces; the prototypes are in the
    network, so this objective holds nothing of its own.

    Attributes:
        margin: The angle, in radians, the true class is pushed away by before scoring.
        scale: What the cosines are multiplied by. Cross-entropy over values bounded by ±1 can never
            become confident, so the softmax needs a temperature; the customary value is this one.
    """

    def __init__(self, margin: float = 0.5, scale: float = 64.0) -> None:
        if not 0.0 <= margin < math.pi:
            raise ValueError(f"An angular margin is an angle in radians, under half a turn; it was {margin}.")
        if scale <= 0:
            raise ValueError(f"The scale is what lets a bounded score become confident; it was {scale}.")
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        """The objective, taken in single precision whatever precision the run trains in.

        Autocast is disabled rather than the input merely cast, because it patches ``matmul`` itself and
        would take the prototype comparison back down to half. See ``COSINE_LIMIT`` for what that costs.
        """
        labels = self._labels(targets)
        with torch.autocast(device_type=outputs.device.type, enabled=False):
            penalized = self._penalized(self._cosines(outputs.float()), labels)
            return self.reported(cross_entropy(penalized * self.scale, labels))

    def _labels(self, targets: Tensor) -> Tensor:
        """Which identity each sample is, as one index per sample.

        Refused rather than read as it stands: a batch transform that blends two samples leaves a share
        of each identity, and ``.long()`` would flatten that to zeros while broadcasting let
        cross-entropy accept the result — two different mixes reporting the very same number.
        """
        if targets.is_floating_point() or targets.ndim != 1:
            raise ValueError(
                f"{type(self).__name__} adds its margin to the one identity a sample is, so it reads one "
                f"index per sample; it was handed {tuple(targets.shape)} of {targets.dtype}. A transform "
                "that mixes two samples leaves a share of each identity instead, and an angular margin "
                "is not defined over that: drop the transform, or the objective."
            )
        return targets.long()

    def _cosines(self, outputs: Tensor) -> Tensor:
        """What the head already produced, held to what a cosine can be.

        Checked every step rather than at build, because no declaration says what a head's values mean.
        A plain projection would otherwise train: every angle saturates at the bound, the margin means
        nothing, and the reported number goes on looking like an objective doing its work.
        """
        largest = float(outputs.detach().abs().amax()) if outputs.numel() else 0.0
        if largest > COSINE_CEILING:
            raise ValueError(
                f"{type(self).__name__} scores the angle between a sample and each identity, so it reads "
                f"cosines; what it was handed reaches {largest:.3g}. Declare `head: cosine` to compare "
                "against prototypes the network holds, or `loss: arcface_proxy` to hold them here instead."
            )
        return outputs

    def _penalized(self, cosines: Tensor, labels: Tensor) -> Tensor:
        """Every cosine as it stands, except the true identity's, which is scored from further away.

        Past half a turn the cosine rises again, so a sample already pointing away from its identity
        would be *rewarded* by the margin; beyond that point the published form falls back to a linear
        penalty. Measured over 400 angles at ``margin=0.5``: written as ``cos(angle + margin)`` alone the
        score turns back upward at 2.638 rad — ``pi - margin`` — while with the fallback it decreases
        throughout.
        """
        angle = cosines.clamp(-COSINE_LIMIT, COSINE_LIMIT).arccos()
        pushed = torch.where(
            angle + self.margin < math.pi,
            (angle + self.margin).cos(),
            cosines - self.margin * math.sin(self.margin),
        )
        return torch.where(one_hot(labels, cosines.size(FEATURE_AXIS)).bool(), pushed, cosines)


@loss_registry.register("arcface_proxy")
class ArcFaceProxy(ArcFace):
    """The same margin, over one prototype per identity that this objective holds and learns itself.

    They are parameters of the run rather than of the network: optimized with it, written to its
    checkpoints, restored with the epoch it keeps, and absent from everything it ships.
    """

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__(margin=margin, scale=scale)
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)

    @override
    def _cosines(self, outputs: Tensor) -> Tensor:
        """The angle between each embedding and each prototype, both read as directions alone.

        Normalizing here rather than expecting it of the model keeps both sides of the comparison in one
        place: what a task publishes is its own business, and so is what this compares.

        The prototypes are cast alongside what arrives, because ``precision=16-true`` casts parameters
        too and upcasting one side alone is a refusal to multiply rather than a loss of precision.
        """
        return normalize(outputs, dim=FEATURE_AXIS) @ normalize(self.prototypes.float(), dim=-1).T
