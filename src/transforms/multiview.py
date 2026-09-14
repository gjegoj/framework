"""Several draws of one sample, stacked: what an objective comparing a picture with itself reads."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.core import Geometry, Modality, Sample, require_tensor
from src.transforms.albumentations import AlbumentationsTransform
from src.transforms.base import GeometryAware, SampleTransform


class MultiViewTransform:
    """One input drawn several times and stacked on a leading axis, for an objective that compares them.

    The view axis rides with the batch axis rather than entering any declared shape. What an input *is*
    — this many channels at this size — is what one view is, and it is what a deployed artifact takes;
    views are a device of training, like the batch axis itself, which ``TensorShape`` deliberately does
    not carry either. So only two places know a run has views at all: this transform, which makes them,
    and the backbone that folds them back into the batch before encoding.

    Every view is the stage's own chain run again on the sample as it was loaded. A chain that draws
    therefore gives each view its own draw, and one that does not gives identical copies — which is a
    declaration saying there is nothing here to learn rather than a case worth guarding.

    Exactly one value may move with the pixels, and anything else that does is refused rather than
    arranged for. The legacy this replaces drew the whole sample per view and kept only the picture, so
    a segmentation target stayed where it was loaded while every view moved away from it — the very
    disagreement ``GeometryAware`` exists to prevent.

    Parameters:
        views: How many draws of the input one sample becomes; at least two, since they are compared.
        base: The stage's own chain, run once per view.
        input_name: Which of the sample's inputs is drawn several times.
    """

    def __init__(self, views: int, base: SampleTransform | GeometryAware, input_name: str = Modality.IMAGE) -> None:
        if views < 2:
            raise ValueError(
                f"Views are read against one another, so a run declares at least two; this one declares "
                f"{views}, which leaves nothing on the other side of the comparison."
            )
        self.views = views
        self.base = base
        self.input_name = input_name

    def with_geometry(
        self, inputs: Mapping[str, Geometry], targets: Mapping[str, Geometry], auxiliary_inputs: Mapping[str, Geometry]
    ) -> SampleTransform:
        """The drawing this declaration becomes, once it is settled that only the viewed input moves.

        A second object rather than a copy of this one, which is the shape the chain below already
        takes: what a run declares and what a stage calls are different things, and a declaration that
        was never bound is not callable at all.
        """
        self._refuse_a_value_that_would_be_left_behind(inputs, targets, auxiliary_inputs)
        self._refuse_a_chain_whose_draw_is_an_answer()
        base = self.base
        bound = base.with_geometry(inputs, targets, auxiliary_inputs) if isinstance(base, GeometryAware) else base
        return DrawnViews(self.views, bound, self.input_name)

    def _refuse_a_value_that_would_be_left_behind(
        self, inputs: Mapping[str, Geometry], targets: Mapping[str, Geometry], auxiliary_inputs: Mapping[str, Geometry]
    ) -> None:
        """A value moving with the pixels is drawn per view too, and a sample has one place to keep it."""
        moving = sorted(
            name
            for declared in (inputs, targets, auxiliary_inputs)
            for name, geometry in declared.items()
            if geometry is not Geometry.NONE
        )
        if moving != [self.input_name]:
            raise ValueError(
                f"Views are drawn of {self.input_name!r}, and {', '.join(moving) or 'nothing'} moves with the "
                f"pixels in this run: a value drawn beside a view has as many versions as there are views "
                f"and one place to be kept, so every view but one would disagree with it. Declare views "
                f"over a run whose only pixels are the input being viewed."
            )

    def _refuse_a_chain_whose_draw_is_an_answer(self) -> None:
        """An augmentation whose draw is the supervision writes one answer, and views need one each."""
        if isinstance(self.base, AlbumentationsTransform) and (answered := sorted(self.base.answers)):
            raise ValueError(
                f"This chain answers {', '.join(answered)} by what it draws, and it is drawn once per view: "
                f"the answers would differ between views while a sample carries one. Declare the views over "
                f"a chain that only augments, and let the pretext task have a stage of its own."
            )


class DrawnViews:
    """A bound drawing: the stage's chain, already tied to what moves, run once per view and stacked.

    Picklable, as the bound chain it wraps is, because a loader worker is a process and this is what it
    carries. Everything but the viewed input comes from the sample as it arrived — nothing else moves
    with the pixels, which is what the declaration settled before building this.
    """

    def __init__(self, views: int, base: SampleTransform, input_name: str) -> None:
        self.views = views
        self.base = base
        self.input_name = input_name

    def __call__(self, sample: Sample) -> Sample:
        drawn = [self.base(sample) for _ in range(self.views)]
        views = torch.stack([require_tensor(one.inputs[self.input_name], name=self.input_name) for one in drawn])
        return Sample(
            inputs={**sample.inputs, self.input_name: views},
            targets=sample.targets,
            metadata=sample.metadata,
            auxiliary_inputs=sample.auxiliary_inputs,
        )
