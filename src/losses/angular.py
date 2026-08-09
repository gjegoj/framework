"""Angular-margin criteria: embeddings learned from class labels (the ArcFace family)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from torch.nn import functional

from src.core.entities import Loss
from src.core.ports import Criterion
from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor

COSINE_TOLERANCE = 1.001
"""How far past [-1, 1] a logit may sit before it is clearly not a cosine."""


class ArcFaceLoss(nn.Module):
    """Additive angular margin on cosine logits, then cross-entropy.

    The target class must win by ``margin`` radians, not merely win —
    ``cos(θ_y + m)`` replaces its logit — which is what forces tight classes
    and wide gaps between them. Past ``π - m`` the substitution stops being
    monotone, so a linear penalty takes over there (the paper's
    ``easy_margin=False``).

    ``margin`` and ``scale`` are plain numbers read anew on every step, so the
    ``anneal`` callback can warm the margin up — the usual way to keep early
    training from collapsing.

    Parameters:
        margin (float): Additive angular margin in radians.
        scale (float): Multiplier restoring logit magnitude after the cosine
            squashed it into [-1, 1].

    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition" (2019).
    """

    def __init__(self, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        if not 0.0 <= margin < math.pi:
            raise ValueError(f"ArcFace margin is an angle in [0, pi) radians, got {margin}.")
        if scale <= 0:
            raise ValueError(f"ArcFace scale must be positive, got {scale}.")
        self.margin = margin
        self.scale = scale

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(f"ArcFace expects cosine logits of shape [B, C], got {tuple(logits.shape)}.")
        if bool(logits.abs().amax() > COSINE_TOLERANCE):
            raise ValueError(
                "ArcFace expects cosine logits in [-1, 1], and these reach "
                f"{logits.abs().amax().item():.2f} — raw scores, not cosines. Use it as the "
                "'inner' of arcface_proxy, which produces cosines from an embedding."
            )
        # Derived per step rather than cached at construction, so an annealed margin acts.
        cos_margin, sin_margin = math.cos(self.margin), math.sin(self.margin)
        threshold = math.cos(math.pi - self.margin)
        linear_penalty = math.sin(math.pi - self.margin) * self.margin

        cosine = logits.clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine**2).clamp_min(0.0))
        with_margin = cosine * cos_margin - sine * sin_margin  # cos(θ + m)
        with_margin = torch.where(cosine > threshold, with_margin, cosine - linear_penalty)

        labels = target.long()
        is_target = functional.one_hot(labels, cosine.shape[1]).bool()
        return functional.cross_entropy(torch.where(is_target, with_margin, cosine) * self.scale, labels)


@criterion_registry.register("arcface")
class ArcFaceCriterion(WrappedCriterion):
    """ArcFace as a criterion — see :class:`ArcFaceLoss` for the math.

    Pairs with the ``cosine`` head, which owns the class prototypes the model
    deploys::

        tasks:
          person:
            preset: classification
            target: person_id
            head: {name: cosine}
            loss: {name: arcface, margin: 0.3}

    Parameters:
        **kwargs: Forwarded verbatim to :class:`ArcFaceLoss`
            (``margin``, ``scale``).
    """

    part_name: ClassVar[str] = "arcface"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ArcFaceLoss(**kwargs))


@criterion_registry.register("arcface_proxy")
class ProxyAngularCriterion(Criterion):
    """Learnable class prototypes turning an embedding head into a cosine classifier.

    Train the embedder through proxy classification, then deploy it without the
    proxies: the prototypes are deliberately part of the *criterion*, so they
    train and checkpoint with the run but never enter the exported model.

    **Which of the two to declare is decided by the export boundary.** An *embedder*
    (faces, retrieval) throws the prototypes away after training, so it wants this one.
    A *classifier* with ArcFace geometry needs them at inference, so a ``cosine`` head
    owns them in the model and ``arcface`` is the stateless margin over its logits.

    ``num_classes`` and ``embedding_dim`` are never written in config — assembly
    offers them, from the fitted labels and from the stream the task reads. The
    loss logs under the inner criterion's name, so swapping the margin renames
    the logged part honestly.

    Parameters:
        num_classes (int): One prototype per class of the fitted vocabulary.
        embedding_dim (int): Width of the embedding the head produces.
        inner (Criterion | None): The margin criterion the cosine logits go to;
            ``None`` builds the default ArcFace from the remaining arguments.
        **kwargs: Forwarded verbatim to the default :class:`ArcFaceCriterion`.
    """

    def __init__(self, num_classes: int, embedding_dim: int, inner: Criterion | None = None, **kwargs: Any) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"Proxy classification needs at least two classes, got {num_classes}.")
        if embedding_dim < 1:
            raise ValueError(f"An embedding needs a positive width, got {embedding_dim}.")
        if inner is not None and kwargs:
            raise ValueError(
                f"arcface_proxy takes either an 'inner' criterion or arguments for the default "
                f"arcface, not both; declare {sorted(kwargs)} on the module itself."
            )
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self._margin = inner if inner is not None else ArcFaceCriterion(**kwargs)

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        # ``logits`` carries the [B, D] embedding: for a metric task the head is identity.
        if logits.ndim != 2 or logits.shape[1] != self.prototypes.shape[1]:
            raise ValueError(
                f"Proxy prototypes are {self.prototypes.shape[1]}-dimensional but the head produced "
                f"{tuple(logits.shape)}; the embedding and the prototypes must share a width."
            )
        cosine = functional.linear(functional.normalize(logits, dim=1), functional.normalize(self.prototypes, dim=1))
        loss: Loss = self._margin(cosine, target)
        return loss
