"""MixUp / CutMix batch transforms (soft-blend / hard-patch label mixing).

Both produce *soft* class targets ``[B, C]`` for every GLOBAL head, sharing one
sampled mix across the image and all heads (see ``multihead``). They support only
GLOBAL topologies: a blended/patched image has no coherent dense (segmentation)
target, so the compatibility guard rejects them when a DENSE head is present.
"""

from __future__ import annotations

from src.core.entities import Batch
from src.core.keys import IMAGE
from src.tasks.taxonomy import Topology
from src.transforms.batch.multihead import CutMixMultiHead, MixUpMultiHead, _MultiHeadMix
from src.transforms.batch.registry import batch_transforms
from src.transforms.batch.spec import BatchTransform, TargetSpec


class _LabelMixTransform(BatchTransform):
    """Shared base: mix the image and rewrite every GLOBAL target as a soft label.

    The concrete mixer (MixUp vs CutMix) is injected by the subclass, so the base never
    calls back into a not-yet-initialised subclass during construction.

    Parameters:
        targets (list[TargetSpec]): Every task whose target must be rewritten (all
            GLOBAL — guaranteed by the compatibility guard).
        mixer (_MultiHeadMix): The multi-head image+label mixer to apply.
        input_key (str): ``Batch.inputs`` key (image stream) to mix.
    """

    supported_topologies: frozenset[Topology] = frozenset({Topology.GLOBAL})

    def __init__(self, targets: list[TargetSpec], mixer: _MultiHeadMix, input_key: str = IMAGE) -> None:
        self._targets = targets
        self._input_key = input_key
        self._mixer = mixer

    def __call__(self, batch: Batch) -> Batch:
        image = batch.inputs[self._input_key]
        labels = {spec.key: batch.targets[spec.key] for spec in self._targets}
        num_classes = {spec.key: spec.num_classes for spec in self._targets}
        mixed_image, mixed_labels = self._mixer.mix(image, labels, num_classes)
        return Batch(
            inputs={**batch.inputs, self._input_key: mixed_image},
            targets={**batch.targets, **mixed_labels},
            meta=batch.meta,
        )


@batch_transforms.register("mixup")
class MixUp(_LabelMixTransform):
    """MixUp: blends two images and their labels by a Beta-sampled weight."""

    def __init__(self, targets: list[TargetSpec], input_key: str = IMAGE, alpha: float = 1.0) -> None:
        super().__init__(targets, MixUpMultiHead(alpha=alpha), input_key=input_key)


@batch_transforms.register("cutmix")
class CutMix(_LabelMixTransform):
    """CutMix: pastes a patch of one image onto another, mixing labels by area."""

    def __init__(self, targets: list[TargetSpec], input_key: str = IMAGE, alpha: float = 1.0) -> None:
        super().__init__(targets, CutMixMultiHead(alpha=alpha), input_key=input_key)
