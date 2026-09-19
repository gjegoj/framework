"""Learning from a second network: how far a student's answer is from a teacher's, as one number."""

from __future__ import annotations

from typing import override

from torch import Tensor
from torch.nn.functional import kl_div, log_softmax

from src.core import FEATURE_AXIS, LossOutput, Representation
from src.losses.base import Loss
from src.losses.registry import distillation_loss_registry


@distillation_loss_registry.register("kullback_leibler")
class KullbackLeibler(Loss):
    """The divergence between a student's answer and a teacher's, both softened before they are compared.

    What the second network buys is everything the targets leave out. A label says one class is right;
    a trained teacher says how wrong each of the others is, and those proportions are what a small
    network cannot work out from few examples. So this is a divergence between two whole distributions
    rather than a second look at the true class.

    Both are softened because the distinctions worth learning sit in the small probabilities, and an
    unsoftened teacher spends all of its confidence on one class. Softening also shrinks the term, which
    is why it is scaled back by the square of the temperature: measured on random logits, without that
    factor the gradient falls fourfold per doubling of the temperature, so the share of the objective
    this is worth would silently mean less at every step of it.

    Parameters:
        temperature: How far both distributions are softened before they are compared. One compares them
            as they are; the usual range is two to ten.
        scale: What a bounded answer is multiplied by before any of the above. Left out for a head whose
            answer is already a logit, which is the common case and the default.
    """

    def __init__(self, temperature: float = 4.0, scale: float | None = None) -> None:
        """Refuse the two numbers that would leave nothing to compare, where they are written.

        A scale is how a cosine becomes a logit, so declaring one declares that this reads cosines —
        that pair is settled here rather than by a third knob repeating what the head already states.

        It is asked for rather than taken from the angular objective beside it, and not for want of a
        way to reach it. Measured on eight real classes: at the customary ``arcface`` scale of 16 a
        teacher's answer softens to 1.0000 with an entropy of 0.0000, which is a hard label written the
        long way, while the objective descending those very cosines is working exactly as intended. The
        sharpness that leaves something to learn is its own choice — two to four classes still standing
        — and borrowing the objective's would look like one fact with one home and be two.
        """
        if temperature <= 0:
            raise ValueError(
                f"A temperature softens a distribution by dividing by it, so it is positive; got {temperature}."
            )
        if scale is not None and scale <= 0:
            raise ValueError(f"A scale is what lets an answer bounded by ±1 become a distribution at all; got {scale}.")
        super().__init__()
        self.temperature = temperature
        self.scale = 1.0 if scale is None else scale
        self.reads = Representation.PROJECTED if scale is None else Representation.COSINES

    @override
    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        """``targets`` is the teacher's answer to the same batch, in the same space as ``outputs``.

        A soft target is still a target, which is why this is a loss like any other and reads the second
        argument every loss reads. Summed over the classes and averaged over everything else, so that
        the number means the same for a task answering once per sample and one answering once per pixel:
        for a flat output that is exactly torch's ``batchmean``, and for a dense one ``batchmean`` would
        report the sum over an image.
        """
        softened = log_softmax(outputs * self.scale / self.temperature, dim=FEATURE_AXIS)
        teaching = log_softmax(targets * self.scale / self.temperature, dim=FEATURE_AXIS)
        pointwise = kl_div(softened, teaching, reduction="none", log_target=True)
        return self.reported(pointwise.sum(FEATURE_AXIS).mean() * self.temperature**2)
