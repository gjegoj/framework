"""Identities: which thing a picture is of, learned from the training split rather than declared.

The one vocabulary in this framework that is *not* declared, and the reason is that it is never
published. What a metric-learning run answers with is a direction; the identities are the device that
teaches the encoder to point, and no position of any output is indexed by them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Self, override

import torch
from torch import Tensor

from src.core import Distribution, TargetInfo
from src.data.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import counted

log = logging.getLogger(__name__)


@target_encoder_registry.register("identity")
class IdentityEncoder(TargetEncoder):
    """Which identity a sample is, as one index; the training split settles how many there are.

    Declared vocabularies are the rule here — ``Encoder`` says so — and this is the exception, named
    where the rule is. What makes it one is that this vocabulary leaves no trace outside the run: the
    shipped record writes no classes for a task that answers with a direction, and the objective's one
    prototype per identity is a parameter of the run rather than of the network. Nothing a deployment
    reads is indexed by these numbers, so the reason declarations exist — that position 0 must mean the
    same thing in every run — has nothing to bite on.

    Sizing the objective from a *declaration* instead is what this replaces, and it is wrong in a way
    that only shows on an open split: measured on Oxford-IIIT Pet with 37 declared breeds and 26 in the
    training split, eleven prototypes were never any sample's positive — training only ever pushed them
    away — and the validation objective over them rose from 43.6 to 60.0 across four epochs while the
    model improved. The checkpoint monitoring it kept the epoch before any training, and the run shipped
    that.

    An identity the training split never showed is not refused: an evaluation split holding its own
    identities is exactly what this kind of task is measured on. Such an identity is interned while its
    split is read, past the learned ones, so a reading comparing identities for equality can tell two
    unseen ones apart without ever mistaking either for a learned one.
    """

    def __init__(self) -> None:
        self._learned: dict[str, int] = {}
        self._met: dict[str, int] = {}

    @property
    def info(self) -> TargetInfo:
        """The identities the training split settled, and the fact that a split may hold others.

        What was interned while another split was read is deliberately absent: an objective is sized
        from this, and it can only learn what it was trained on.
        """
        learned = self._require_learned()
        return TargetInfo(classes={index: name for name, index in learned.items()}, open_set=True)

    def fit(self, values: Iterable[object]) -> Self:
        """Every identity of the training split, indexed in sorted order.

        Sorted rather than first seen, and it is the objective that makes this load-bearing: it keeps one
        prototype per index, writes them to every checkpoint and restores them by index. An order that
        followed the rows would move with the split seed, the row order and ``max_samples``, and restored
        prototypes would then be paired with identities they were never learned for.
        """
        names = _named(values)
        if not names:
            raise ValueError(f"{type(self).__name__} learns the identities of the training split, and it is empty.")
        self._learned = {name: index for index, name in enumerate(sorted(set(names)))}
        self._met = {}
        log.info("%s learned %d identities from the training split.", type(self).__name__, len(self._learned))
        return self

    @override
    def validate(self, values: Iterable[object]) -> None:
        """Take in whatever this split names, rather than hold it to what training showed.

        The opposite answer to ``LabelEncoder``'s, to the same question, and the difference is what an
        evaluation split is *for* here: identities the run never trained on are how open-set retrieval is
        measured at all.

        This is the one encoder whose ``validate`` leaves the encoder able to encode more than before.
        The line it does not cross is ``info``: what a split adds is never published, so nothing sized
        from this target can be sized from a split the run only reads.
        """
        for name in _named(values):
            if name not in self._require_learned() and name not in self._met:
                self._met[name] = len(self._learned) + len(self._met)

    def encode(self, value: object) -> Tensor:
        """The index this identity was given while its split was read.

        Refused rather than given one here: encoding happens once per sample, in whichever worker holds
        a copy of this encoder, and two workers inventing an index for the same unread identity would
        invent two — one thing counted as two different things by every reading that compares them.
        """
        name = _one_name(value)
        index = self._require_learned().get(name, self._met.get(name))
        if index is None:
            raise LookupError(
                f"Identity {name!r} is in no split this run read. The identities of the training split "
                "are learned when it is fitted, and every other prepared split is read before the first "
                "batch; one arriving later belongs to neither."
            )
        return torch.tensor(index, dtype=torch.long)

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        """One count per identity, seeded with the learned ones so a split shows which of them it holds.

        Which is the picture of an open split doing its job: the identities of training read zero here,
        and the ones this split brought stand beside them as rows of their own.
        """
        return counted({index: name for name, index in self._require_learned().items()}, _named(values))

    def _require_learned(self) -> dict[str, int]:
        """The learned identities, or a refusal naming the two ways a run can have them.

        Reached from ``info``, which the composition root asks for before the model exists — so a run
        that prepares no training split is refused while it is assembled rather than on its first batch.
        """
        if not self._learned:
            raise ValueError(
                f"{type(self).__name__} learns its identities from the training split, and this run "
                "prepares none. Train, or declare `tasks.<name>.target_encoder: label` together with "
                "`tasks.<name>.classes` to pin the vocabulary so nothing needs learning."
            )
        return self._learned


def _named(values: Iterable[object]) -> list[str]:
    return [_one_name(value) for value in values]


def _one_name(value: object) -> str:
    """One identity as the column wrote it, stripped; a cell naming none is refused where it is read."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
        raise ValueError(
            "A cell names no identity, and whether those rows are one thing or none of them is a "
            "question about the data. Give them an identity, or leave them out of the table."
        )
    return str(value).strip()
