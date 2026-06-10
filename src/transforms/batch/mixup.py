"""MixUp / CutMix batch transforms (soft-blend / hard-patch label mixing).

Both produce *soft* class targets ``[B, C]`` for every GLOBAL head, sharing one
sampled mix across the image and all heads (see ``multihead``). They support only
GLOBAL topologies: a blended/patched image has no coherent dense (segmentation)
target, so the compatibility guard rejects them when a DENSE head is present.
"""

from __future__ import annotations

from src.core.entities import Batch
from src.core.keys import IMAGE
from src.core.ports import BatchTransform
from src.tasks.taxonomy import Topology
from src.transforms.batch.multihead import CutMixMultiHead, MixUpMultiHead, _MultiHeadMix
from src.transforms.batch.registry import batch_transforms
from src.transforms.batch.spec import TargetSpec


class _LabelMixTransform(BatchTransform):
    """Shared base: mix the image and rewrite every GLOBAL target as a soft label.

    Parameters:
        targets (list[TargetSpec]): Every task whose target must be rewritten (all
            GLOBAL — guaranteed by the compatibility guard).
        input_key (str): ``Batch.inputs`` key (image stream) to mix.
        alpha (float): Beta-distribution parameter controlling the mix strength.
    """

    supported_topologies: frozenset[Topology] = frozenset({Topology.GLOBAL})

    def __init__(self, targets: list[TargetSpec], input_key: str = IMAGE, alpha: float = 0.2) -> None:
        self._targets = targets
        self._input_key = input_key
        self._mixer = self._build_mixer(alpha)

    def _build_mixer(self, alpha: float) -> _MultiHeadMix:
        raise NotImplementedError

    def __call__(self, batch: Batch) -> Batch:
        image = batch.inputs[self._input_key]
        labels = {t.key: batch.targets[t.key] for t in self._targets}
        num_classes = {t.key: t.num_classes for t in self._targets}
        mixed_image, mixed_labels = self._mixer.mix(image, labels, num_classes)
        return Batch(
            inputs={**batch.inputs, self._input_key: mixed_image},
            targets={**batch.targets, **mixed_labels},
            meta=batch.meta,
        )


@batch_transforms.register("mixup")
class MixUp(_LabelMixTransform):
    """MixUp: blends two images and their labels by a Beta-sampled weight."""

    def _build_mixer(self, alpha: float) -> _MultiHeadMix:
        return MixUpMultiHead(alpha=alpha)


@batch_transforms.register("cutmix")
class CutMix(_LabelMixTransform):
    """CutMix: pastes a patch of one image onto another, mixing labels by area."""

    def _build_mixer(self, alpha: float) -> _MultiHeadMix:
        return CutMixMultiHead(alpha=alpha)
