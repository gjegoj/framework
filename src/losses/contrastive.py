"""Contrastive loss criteria for MULTISTREAM topology.

Receives ``logits: [B, 2, D]`` — two embedding streams per sample from two
separate encoders — and aligns them: row ``i`` of stream 0 matches row ``i`` of
stream 1 (the diagonal), every other pairing is a negative.  The ``target`` is
ignored; supervision is implicit in the pairing.

The ``[B, N, D]`` carrier is N-general, so an N-way variant can be added later
without upstream changes.  ``PairedStreamCriterion`` is the family's extension
point — it lives here rather than in ``base.py`` because its only consumers are
this family's losses (cross-family bases go to ``base``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.base import require_view_shape
from src.losses.registry import criteria

# CLIP clamps the exponentiated logit scale to keep training stable.
_MAX_LOGIT_SCALE = 100.0


class PairedStreamCriterion(Criterion):
    """Base for contrastive losses over two L2-normalized streams ``[B, 2, D]``.

    Holds the learnable log-space ``logit_scale`` and the preparation shared by
    every pairwise objective — shape validation, per-stream L2-normalization, and
    the clamped multiplicative scale. Subclasses initialise ``logit_scale`` in
    ``__init__`` and implement the pairwise objective in ``forward``.
    """

    logit_scale: nn.Parameter

    def _normalized_streams(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Validate ``[B, 2, D]`` and return ``(stream0, stream1, scale)``, streams L2-normalized."""
        require_view_shape(logits, views=2, owner=type(self).__name__)
        anchor = F.normalize(logits[:, 0], dim=-1)
        other = F.normalize(logits[:, 1], dim=-1)
        scale = self.logit_scale.exp().clamp(max=_MAX_LOGIT_SCALE)
        return anchor, other, scale


@criteria.register("info_nce")
class InfoNCECriterion(PairedStreamCriterion):
    """Symmetric InfoNCE (CLIP loss) on ``[B, 2, D]`` embeddings.

    L2-normalizes each stream, forms the ``[B, B]`` similarity matrix scaled by a
    learnable temperature, and averages the two cross-entropies (stream0→stream1
    and stream1→stream0) against the diagonal targets.

    Parameters:
        temperature (float): Initial softmax temperature.  Stored as a learnable
            log-space ``logit_scale`` parameter (initialised to ``log(1/temperature)``),
            exactly like CLIP, so it adapts during training.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        anchor, other, scale = self._normalized_streams(logits)
        similarity = scale * anchor @ other.t()  # [B, B]
        labels = torch.arange(similarity.size(0), device=similarity.device)

        loss_anchor = F.cross_entropy(similarity, labels)
        loss_other = F.cross_entropy(similarity.t(), labels)
        value: Tensor = (loss_anchor + loss_other) / 2.0
        return LossResult(total=value, components={"info_nce": value})


@criteria.register("siglip")
class SigLIPCriterion(PairedStreamCriterion):
    """Sigmoid pairwise loss (SigLIP) on ``[B, 2, D]`` embeddings.

    Unlike InfoNCE's batch-normalized softmax, SigLIP treats every cell of the
    ``[B, B]`` similarity matrix as an independent binary problem: the diagonal
    pairs are positives (+1), all others negatives (-1).  This needs no
    cross-batch normalization, so it scales to large batches.  Two parameters are
    learnable — the logit ``scale`` *and* a ``bias`` — both initialised as in the
    paper (scale ``10``, bias ``-10``), which places the initial logits at the
    decision boundary.

    Parameters:
        logit_scale (float): Initial multiplicative logit scale ``t``; stored as a
            learnable log-space parameter, exactly like CLIP/SigLIP.
        bias (float): Initial additive logit bias ``b`` (learnable).

    Reference:
        Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (2023).
    """

    def __init__(self, logit_scale: float = 10.0, bias: float = -10.0) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        anchor, other, scale = self._normalized_streams(logits)
        pair_logits = scale * anchor @ other.t() + self.bias  # [B, B]

        # +1 on the diagonal (positive pairs), −1 elsewhere (negatives).
        batch = pair_logits.size(0)
        signs = 2.0 * torch.eye(batch, device=pair_logits.device) - 1.0
        value: Tensor = -F.logsigmoid(signs * pair_logits).sum() / batch
        return LossResult(total=value, components={"siglip": value})
