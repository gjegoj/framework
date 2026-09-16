"""A target that is an identity rather than a class, and an answer that is a direction rather than a score."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, override

from torch import Tensor
from torch.nn.functional import normalize

from src.core import FEATURE_AXIS, Axis, Batch, Representation, TargetInfo, TensorTree
from src.tasks.base import LossDeclaration, TargetEncoderDeclaration, Task
from src.tasks.registry import task_registry


@task_registry.register("metric_learning")
class MetricLearning(Task):
    """Samples of one identity are learned to point the same way, so unseen identities can be told apart.

    A classifier can only answer about the vocabulary it was trained on. This kind is declared where the
    identities at deployment are not the ones at training — a face, a product, a re-identified vehicle —
    and what it produces is a direction that any gallery can be searched with by angle.

    Three things follow, and they are what make this kind unlike the others. The width of the answer is a
    choice of the *model* rather than something the data settles, so it is declared on the kind and
    published as a fact for whoever is sized from it. It has no semantics: the vocabulary is a training
    device rather than the meaning of the output, so a run declaring metrics that score against one is
    refused at build by the library they come from — rather than reporting a plausible-looking f1 over
    identities the model never answers about.

    And that vocabulary is the one in this framework that is *learned* rather than declared, which is
    what lets a run hold identities out of training and measure the thing the method exists for. The
    objective is then sized by the identities it can actually learn, and an evaluation split naming its
    own is not held to them. ``IdentityEncoder`` carries the reason and the measurement.
    """

    output_axis: ClassVar[str] = Axis.EMBEDDING
    publishes: ClassVar[Representation] = Representation.DIRECTION
    default_target_encoder: ClassVar[TargetEncoderDeclaration | None] = "identity"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {
        "recall_at_1": {"name": "recall_at_k", "k": 1},
        # Recall asks whether a match turned up first; this asks how every picture of the identity was
        # ranked, which is the reading that separates a model finding one of six from one finding six.
        "map": {"name": "map"},
    }

    def __init__(
        self,
        name: str,
        info: TargetInfo,
        *,
        embedding_dim: int,
        weight: float = 1.0,
        lr: float | None = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError(
                f"'embedding_dim' is how wide the direction this task answers with is, and there is no "
                f"defensible default for it; it was declared as {embedding_dim}."
            )
        if info.num_classes is None:
            raise ValueError(
                f"Task {name!r} is learned by separating identities, one prototype each, so it needs to "
                "know how many there are, and nothing it reads its column with says. Leave "
                "`tasks.<name>.target_encoder` to this kind's own, which learns them from the training "
                "split, or declare `label` together with `tasks.<name>.classes` to pin them."
            )
        super().__init__(name, info, weight=weight, lr=lr)
        self.embedding_dim = embedding_dim

    def out_features(self) -> int:
        return self.embedding_dim

    @override
    def facts(self) -> Mapping[str, object]:
        """The three every task settles, and the one width only this kind knows.

        An objective holding one prototype per identity sizes them from ``embedding_dim``; a declaration
        writing that width a second time is refused before the run starts.
        """
        return {**super().facts(), "embedding_dim": self.embedding_dim}

    @property
    def default_loss(self) -> LossDeclaration:
        """Prototypes in the objective, so a run ships the encoder and its projection and nothing else.

        Keeping them in the network instead — `head: cosine` with `loss: arcface` — makes the artifact
        a classifier over the identities it was trained on. That is a deployment choice, so it is
        declared rather than defaulted to.
        """
        return "arcface_proxy"

    def loss_target(self, batch: Batch) -> Tensor:
        """Which identity the sample is, as one index; whether a share of two may stand here is the
        objective's to declare."""
        return self.target(batch).long()

    def metric_view(self, batch: Batch) -> Tensor:
        """The same index: a retrieval reading asks whether a neighbour turned out to be the same one."""
        return self.target(batch).long()

    @override
    def publish(self, projected: Tensor) -> TensorTree:
        """A unit vector: only the direction carries the identity, and this is what the artifact emits."""
        return normalize(projected, dim=FEATURE_AXIS)
