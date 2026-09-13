"""An artifact is proven by running it beside the model it was written from."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import torch
from torch import Tensor

from src.export.base import BATCH_AXIS, Exporter
from src.export.deployable import DeployableModel, as_outputs


@dataclass(frozen=True, slots=True)
class Parity:
    """How far a written artifact stands from the model, and the batch sizes it was asked at.

    ``allowance_used`` is the share of the allowance the worst value consumed, so one quantity answers
    both of a reader's questions — how close, and close enough — and a report cannot show one number while judging
    by another. The reference implementation showed ``|d| / (|ref| + 1e-8)`` and judged by
    ``|d| <= atol + rtol*|ref|``, which let a line read 3e+04 under a passing verdict.

    ``difference`` is that same disagreement in the values' own units, because that is what a person
    reads. It is the gap *at that element* rather than the largest gap anywhere, and the two are not the
    same wherever the allowance varies with the value: a relative tolerance is wide where the value is
    large, so the biggest gap can be well inside its allowance while a tiny one elsewhere is far outside.
    Reported as two separate maxima, the refusal reads "disagrees by X, which is Y times what is
    allowed" about no element that exists.
    """

    allowance_used: float
    difference: float
    batches: tuple[int, ...]

    @property
    def within_tolerance(self) -> bool:
        return self.allowance_used <= 1.0


def verify(exporter: Exporter, path: Path, graph: DeployableModel, example: tuple[Tensor, ...]) -> Parity:
    """Run the artifact beside the model on every batch size this format has to serve.

    The oracle is the model itself under ``no_grad``: what an artifact has to be is not some idea of
    correctness but this run's own answers, to the tolerance its format declares.

    The sizes come from the format rather than from constants here — an engine built for a batch profile
    answers inside it and nowhere else — and there is always more than one, because a settled batch axis
    answers the size it was written from perfectly and fails at the single row a deployment sends.

    Raises:
        RuntimeError: If the artifact answers with another shape, another number of outputs, values
            that are not numbers, or values outside the allowance — naming the file and how far out it
            was.
    """
    runnable = exporter.load(path)
    batches = exporter.answers_at(written_at=int(example[0].shape[BATCH_AXIS]))
    used = difference = 0.0
    for batch in batches:
        given = _at_batch(example, batch)
        with torch.no_grad():
            expected = as_outputs(graph(*given))
        written = runnable(given)
        _refuse_an_answer_of_another_shape(path, graph.output_names, written, expected)
        _refuse_an_answer_that_is_not_a_number(path, graph.output_names, written, expected)
        for one, other in zip(written, expected, strict=True):
            gap = (one - other).abs().flatten()
            allowance = exporter.atol + exporter.rtol * other.abs().flatten()
            # Where there is nothing to allow, no allowance was used — said here rather than left to
            # 0/0, whose NaN would then be picked as the worst value and compare false against it.
            share = torch.where(gap > 0, gap / allowance, torch.zeros_like(gap))
            worst = int(share.argmax())
            if float(share[worst]) > used:
                used, difference = float(share[worst]), float(gap[worst])
    parity = Parity(allowance_used=used, difference=difference, batches=batches)
    if not parity.within_tolerance:
        raise RuntimeError(
            f"{path.name} is not the model it was written from: at its worst it disagrees by "
            f"{parity.difference:.3e}, which is {parity.allowance_used:.1f} times what this format allows. The "
            "artifact is written; it is the run that cannot claim it serves this model."
        )
    return parity


def _refuse_an_answer_that_is_not_a_number(
    path: Path, names: tuple[str, ...], written: tuple[Tensor, ...], expected: tuple[Tensor, ...]
) -> None:
    """A value that is not a number is below every bound, because it is below nothing.

    Every comparison that would catch it is therefore false: the share it used stays at its starting
    0.0 and the artifact is reported as standing exactly on the model. It also takes its whole output
    down with it — being the largest share, it is the element the reading is taken from, so a real
    disagreement beside it is recorded as none. Both sides are read, because a model that went unstable
    answers this way too and an artifact faithfully copying it would otherwise agree with it perfectly.
    """
    for name, one, other in zip(names, written, expected, strict=True):
        for source, values in ((path.name, one), ("the model it was written from", other)):
            unusable = int((~torch.isfinite(values)).sum())
            if unusable:
                raise RuntimeError(
                    f"{source} answers {name!r} with {unusable} of {values.numel()} values that are not "
                    "a number, so there is nothing to compare by. A comparison against one is false "
                    "however it is written, which would report this artifact as exactly the model."
                )


def _refuse_an_answer_of_another_shape(
    path: Path, names: tuple[str, ...], written: tuple[Tensor, ...], expected: tuple[Tensor, ...]
) -> None:
    """A disagreement about shape is not a small disagreement, and subtraction would make it look like one.

    ``[batch, 1]`` against ``[batch, 2]`` broadcasts into a difference of ordinary size and a verdict
    that the artifact is the model — which is exactly the reading a settled batch axis produces.
    """
    if len(written) != len(expected):
        raise RuntimeError(
            f"The model answers with {len(expected)} outputs ({', '.join(names)}) and {path.name} with "
            f"{len(written)}, so there is nothing to compare them by."
        )
    for name, one, other in zip(names, written, expected, strict=True):
        if one.shape != other.shape:
            raise RuntimeError(
                f"{path.name} answers {name!r} with shape {tuple(one.shape)} where the model answers "
                f"{tuple(other.shape)}. These do not compare: subtracting them would broadcast into a "
                "difference of ordinary size and a verdict that the artifact is the model."
            )


def _at_batch(example: tuple[Tensor, ...], batch: int) -> tuple[Tensor, ...]:
    """The same example at another batch size, by repeating its rows.

    Repeating rather than drawing fresh values, because drawing would have to know what each input holds
    and repeating knows nothing; either way what is compared is the artifact against the model on the
    very same tensors.
    """
    return tuple(one.repeat(ceil(batch / one.shape[BATCH_AXIS]), *(1,) * (one.ndim - 1))[:batch] for one in example)
