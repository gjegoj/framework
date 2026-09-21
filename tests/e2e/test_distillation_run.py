"""Two networks in one run: one teaches from a file, the other learns — and only one of them survives it.

Assembled through the composition root, because what the phase is about is an arrangement rather than a
formula: which network the optimizer sees, which one a checkpoint holds, and which one is shipped.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from src.build import build
from src.config import load_config
from src.core import submodule_at
from src.experiment import run
from src.models.heads import Mlp
from src.models.weights import weights_in
from src.training import DistillationLearner
from src.training.checkpoints import model_weights
from tests.support.declarations import smallest_run
from tests.support.scripts import cut_head_tail
from tests.support.table import write_table

SOFT = "train/species/distillation"
ALIGNED = "train/pooled/representation"
HIDDEN = "train/species_hidden_0/representation"
"""The term over the head's first hidden width, filed under the name the composite gave that stream."""
MIDDLE = "train/neck_hidden_0/representation"
"""The term over the neck's own first hidden width, filed under the position rather than under a task."""
SHARED = 8
"""The width both networks are brought to, small enough that a head over it is cheap to compare."""
STUDENT = "test_efficientnet"
"""A second family for the student, so that its own width is not its teacher's and a projector has work
to do: measured, this one publishes `pooled` at 256 where the family the fixture declares publishes 96."""
TERMS: list[Mapping[str, Any]] = [
    {"loss": {"name": "kullback_leibler", "temperature": 4.0}},
    {"loss": "mse", "weight": 2.0, "stream": "pooled"},
]
"""Both halves of what a teacher offers: the answers it gives, and the features it read them from."""

HEAD = "heads.species"
"""Where this run's one head sits, in the dot-path a `freeze` declaration names it by."""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def declared(table: Path, tmp_path: Path) -> dict[str, Any]:
    """A run that keeps the epoch it scored best on and writes its numbers down, so both can be read back."""
    return {
        **smallest_run(table, tmp_path),
        "tracker": {"name": "csv", "save_dir": str(tmp_path / "recorded"), "version": ""},
        "callbacks": [{"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(tmp_path / "kept")}],
    }


def teacher_of(declared: Mapping[str, Any], tmp_path: Path) -> str:
    """An ordinary run, whose kept weights are what the run below learns from."""
    ordinary = {**declared, "callbacks": [{"name": "checkpoint", "dirpath": str(tmp_path / "taught")}]}
    run(build(load_config(ordinary)))
    return str(next(iter(sorted((tmp_path / "taught").glob("*.ckpt")))))


def distilling(
    declared: Mapping[str, Any],
    teacher: str,
    loss: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    heads: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A run learning from a second network as well as from its targets; `loss` is how far it is from it."""
    taught = {**declared["model"], "checkpoint_path": teacher}
    return {
        **declared,
        "learner": {
            "name": "distillation",
            "weight": 0.5,
            "loss": loss if loss is not None else {"name": "kullback_leibler", "temperature": 4.0},
            "teacher": taught if heads is None else {**taught, "heads": heads},
        },
    }


def narrowed(declared: Mapping[str, Any], width: int, **declared_neck: Any) -> dict[str, Any]:
    """The same run with its features brought to a width, which is how two networks come to share one.

    A position of its own rather than a backbone wrapped around the backbone: the family stays declared
    by the name it had, the paths a `freeze` and an `adapter` write go on meaning what they meant, and
    nothing here spells out an import path to say which family the fixture chose.

    The rest of the neck's own declaration travels through, because a teacher is built from this very
    mapping: a stack declared here is declared for both networks at once, which is the only way the
    width between what they read and what they publish is the same width in both.
    """
    neck = {"name": "projector", "width": width, **declared_neck}
    return {**declared, "model": {**declared["model"], "neck": neck}}


def of_family(declared: Mapping[str, Any], architecture: str) -> dict[str, Any]:
    """The same run over another architecture, which is what a student and its teacher usually are."""
    backbone = {**declared["model"]["backbone"], "model_name": architecture}
    return {**declared, "model": {**declared["model"], "backbone": backbone}}


def head_of(teacher: str, tmp_path: Path) -> str:
    """The teacher's head, out of its run file, as the tool a reader is pointed at writes it."""
    written = tmp_path / "head.ckpt"
    finished = cut_head_tail(teacher, str(written), "--task", "species")
    assert finished.returncode == 0, finished.stderr
    return str(written)


def head_weights(built: Any) -> dict[str, torch.Tensor]:
    """What the student's head holds, by the dot-path a `freeze` declaration names it by."""
    head = submodule_at(built.module.learner.model, HEAD, reader="test")
    return {name: value.clone() for name, value in head.state_dict().items()}


def answering_through(declared: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
    """The same run, with this head over its one task."""
    return {**declared, "tasks": {"species": {**declared["tasks"]["species"], "head": head}}}


def features_read_by_the_head(declared: Mapping[str, Any]) -> int:
    """How wide the features this run's head reads are, read off a built run rather than written down."""
    head = submodule_at(build(load_config(declared)).module.learner.model, HEAD, reader="test")
    return next(one.in_features for one in head.modules() if isinstance(one, nn.Linear))


def answering_in_angles(declared: Mapping[str, Any]) -> dict[str, Any]:
    """The same run with a `cosine` head under the angular objective that needs one."""
    task = {**declared["tasks"]["species"], "head": {"name": "cosine", "embedding_dim": 8}}
    return {**declared, "tasks": {"species": {**task, "loss": {"name": "arcface", "margin": 0.5, "scale": 16.0}}}}


def test_the_run_keeps_and_ships_the_student_alone(declared: dict[str, Any], tmp_path: Path) -> None:
    """The second network is a way of learning, not a thing the run produces: nothing it leaves holds it."""
    taught = build(load_config(distilling(declared, teacher_of(declared, tmp_path))))
    ordinary = build(load_config(declared))

    run(taught)

    kept = torch.load(next(iter(sorted((tmp_path / "kept").glob("*.ckpt")))), map_location="cpu", weights_only=True)
    # Against what an ordinary run would have written rather than against a name: a second network
    # registered under any word at all is a key here that an ordinary run does not have.
    assert {name for name in kept["state_dict"] if name.startswith("learner.")} == {
        f"learner.{name}" for name in ordinary.module.learner.state_dict()
    }
    assert list(taught.module.learner.model.state_dict()) == list(ordinary.module.learner.model.state_dict())


def test_the_teacher_answers_with_the_weights_it_was_pointed_at(declared: dict[str, Any], tmp_path: Path) -> None:
    """A network whose head was only just initialised answers with noise, and a run would descend towards it."""
    taught = teacher_of(declared, tmp_path)
    distilled = build(load_config(distilling(declared, taught)))
    learner = distilled.module.learner
    assert isinstance(learner, DistillationLearner)

    held = model_weights(taught)
    answering = learner.teacher.state_dict()

    assert set(answering) == set(held)
    assert all(torch.equal(value, held[name]) for name, value in answering.items())
    untrained = build(load_config(declared)).module.learner.model.state_dict()
    assert any(not torch.equal(value, untrained[name]) for name, value in answering.items())


def test_what_the_student_learned_from_the_teacher_is_a_row_of_its_own(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """A term the total is made of and the report does not show is a number nobody can account for."""
    taught = build(load_config(distilling(declared, teacher_of(declared, tmp_path))))

    run(taught)

    written = sorted((tmp_path / "recorded").rglob("metrics.csv"))
    assert len(written) == 1
    with written[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert SOFT in columns


def test_how_far_the_student_is_from_the_teacher_in_features_is_a_row_of_its_own(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """Two terms under one learner, each prefixed by what it read — the task for the answers, the stream
    for the features — because both are part of the total and a term the report hides is a number nobody
    can account for.

    Both networks are the same architecture here, so they publish `pooled` at one width already and no
    projector is wanted: the term compares what two networks both publish, and how each came to that
    width is not its business.
    """

    taught = build(load_config(distilling(declared, teacher_of(declared, tmp_path), loss=TERMS)))

    run(taught)

    written = sorted((tmp_path / "recorded").rglob("metrics.csv"))
    assert len(written) == 1
    with written[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert SOFT in columns
    assert ALIGNED in columns


def test_a_head_that_answers_in_angles_is_distilled_only_where_the_term_is_told_so(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The arrangement this position exists for, and the one it has to refuse — assembled, not simulated.

    A `cosine` head answers in ±1, and a divergence over those numbers is nearly flat whatever the two
    networks think: measured on eight real classes, 256 times smaller than the same comparison made once
    the angles are turned into logits, and every number a run reports about it looks ordinary. Nothing in
    the tensor says which space it holds, so the objective says — and a run that does not say is stopped
    while it is assembled rather than trained on the flat number.
    """
    angular = answering_in_angles(declared)
    teacher = teacher_of(angular, tmp_path)

    with pytest.raises(ValueError, match=r"learner\.loss"):
        build(load_config(distilling(angular, teacher)))

    told = {"name": "kullback_leibler", "temperature": 4.0, "scale": 16.0}
    run(build(load_config(distilling(angular, teacher, told))))

    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        assert SOFT in next(iter(csv.reader(recorded)))


def test_a_student_answers_through_the_head_it_was_given_while_its_teacher_answers_through_its_own(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The arrangement this slice exists for, assembled rather than described.

    A teacher trained with a wide head leaves a stack of layers; the student declares the tail of that
    stack, starts it from a file prepared for exactly those widths, and holds it still. Two networks,
    two different heads, one run — and what the student ends with is the numbers it was handed, which is
    what makes the tail a fixed reading rather than one more thing to learn.
    """
    teacher = teacher_of(answering_through(declared, {"name": "mlp", "hidden_features": [8]}), tmp_path)
    prepared = Mlp(in_features=features_read_by_the_head(declared), out_features=2, hidden_features=[4]).state_dict()
    kept = tmp_path / "tail.pt"
    torch.save(prepared, kept)

    taught = distilling(
        answering_through(declared, {"name": "mlp", "hidden_features": [4], "checkpoint_path": str(kept)}),
        teacher,
        heads={"species": {"name": "mlp", "hidden_features": [8]}},
    )
    built = build(load_config({**taught, "callbacks": [*declared["callbacks"], {"name": "freeze", "modules": [HEAD]}]}))

    run(built)

    held = submodule_at(built.module.learner.model, HEAD, reader="test").state_dict()
    assert all(torch.equal(value, prepared[name]) for name, value in held.items())


def test_a_student_carries_its_teachers_head_over_a_width_they_share_and_it_is_still_the_teachers(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The whole arrangement: a `projector` on both networks brings them to one width, the student's head
    is its teacher's whole, a `freeze` holds it, and the terms pull the features towards the teacher's.

    Held still is what makes the shared space mean anything. Under a divergence over logits alone a
    frozen head constrains nothing — whatever the head, the backbone beneath it is free to make the
    logits come out right — so what is being checked is the pair: the head is the teacher's when the run
    starts, and bit for bit the teacher's when it ends, while the features were what moved.
    """
    shared = narrowed(declared, SHARED)
    teacher = teacher_of(shared, tmp_path)
    carried = weights_in(head_of(teacher, tmp_path))

    student = answering_through(shared, {"name": "linear", "checkpoint_path": head_of(teacher, tmp_path)})
    held = {**student, "callbacks": [*student["callbacks"], {"name": "freeze", "modules": [HEAD]}]}

    built = build(load_config(distilling(held, teacher, loss=TERMS)))

    learner = built.module.learner
    assert isinstance(learner, DistillationLearner)
    taught = learner.teacher
    assert [name for name, _ in taught.named_children()] == ["backbone", "neck", "heads"], (
        "a teacher is a model declaration, so it takes the same neck by the same word and no code in "
        "`training/` knows the position exists"
    )
    kept = model_weights(teacher)
    assert all(
        torch.equal(value, kept[name]) for name, value in taught.state_dict().items() if name.startswith("neck.")
    ), "the teacher's neck was built fresh rather than read from the run that trained it"

    before = head_weights(built)
    assert all(torch.equal(before[name], carried[name]) for name in carried), "the head did not start from the file"

    run(built)

    after = head_weights(built)
    assert all(torch.equal(before[name], after[name]) for name in before), "the frozen head moved"

    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert SOFT in columns
    assert ALIGNED in columns, "a held-still head cannot move whether or not the features were pulled"


def test_a_student_of_another_family_is_brought_to_the_width_its_teacher_already_publishes(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The arrangement that asks nothing of the teacher, and so the cheapest of the three to try: an
    already-trained network is used exactly as it stands, the projector sits on the student alone, and
    what the two come to share is the teacher's own features.

    Two families rather than one, because a projector between equal widths would prove nothing: measured,
    the fixture's publishes `pooled` at 96 and the student's at 256, so the term comparing the two would
    have tensors of different widths and be refused by name were the projector not there.
    """
    teacher = teacher_of(declared, tmp_path)
    shared = features_read_by_the_head(declared)
    carried = weights_in(head_of(teacher, tmp_path))

    taught = distilling(declared, teacher, loss=TERMS)
    mine = narrowed(of_family(declared, STUDENT), shared)["model"]
    carried_by = {"name": "linear", "checkpoint_path": head_of(teacher, tmp_path)}
    student = answering_through({**taught, "model": mine}, carried_by)

    built = build(load_config({**student, "callbacks": [*student["callbacks"], {"name": "freeze", "modules": [HEAD]}]}))
    before = head_weights(built)
    assert all(torch.equal(before[name], carried[name]) for name in carried), "the head did not start from the file"

    run(built)

    assert all(torch.equal(before[name], value) for name, value in head_weights(built).items()), "the head moved"
    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert SOFT in columns
    assert ALIGNED in columns


def test_holding_the_backbone_still_leaves_a_neck_the_run_declared_free_to_learn(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The recipe `examples/finetuning.yaml` ships — hold the pretrained encoder, learn what is above it
    — and the whole reason a projection is a position rather than a backbone wrapped around one.

    With the projection inside the backbone, `modules: [backbone]` held it too: the run trained only its
    heads, the log said `Frozen until …: backbone`, and the space the arrangement exists to learn could
    not move. Here the two are named apart, so one is held and the other learns.
    """
    held = {
        **declared,
        "model": {**declared["model"], "neck": {"name": "projector", "width": SHARED}},
        # `train_bn` off because this test says "held still" and means it: left on, which is the
        # default and the right one for a real finetune, normalisation keeps learning and the encoder's
        # own parameters move — measured here, where it read as a frozen backbone that moved.
        "callbacks": [*declared["callbacks"], {"name": "freeze", "modules": ["backbone"], "train_bn": False}],
    }
    built = build(load_config(held))
    before = {name: value.clone() for name, value in built.module.learner.model.named_parameters()}

    run(built)

    after = dict(built.module.learner.model.named_parameters())
    assert all(torch.equal(before[name], after[name]) for name in before if name.startswith("backbone.")), (
        "the backbone moved while it was held still"
    )
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("neck.")), (
        "the neck never moved, so the space this run exists to learn was frozen with the encoder"
    )


def test_a_students_hidden_widths_are_pulled_towards_its_teachers_by_the_name_the_head_gave_them(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The arrangement whole: a neck on both networks, the teacher's `mlp` head carried and held still,
    and a term over the head's own first hidden width beside the one over the width they share.

    A stream published by the head rather than by the encoding half, named on the term exactly as
    `pooled` is — which is the whole of what the position buys. What can move is the neck, because the
    head cannot: it is bit for bit the file it started from, so every term of this run pulls on the
    features beneath it or on nothing.

    The neck moving is this run's liveness and not the new term's signature — the task's own objective
    reaches it through the held-still head, and `TERMS` already compares `pooled`. What is this term's
    alone is the column: a name no network here published before a head did.
    """
    stacked: dict[str, Any] = {"name": "mlp", "hidden_features": [6, 4]}
    shared = answering_through(narrowed(declared, SHARED), stacked)
    teacher = teacher_of(shared, tmp_path)
    student = answering_through(shared, {**stacked, "checkpoint_path": head_of(teacher, tmp_path)})
    held = {**student, "callbacks": [*student["callbacks"], {"name": "freeze", "modules": [HEAD]}]}
    terms = [*TERMS, {"loss": "mse", "weight": 3.0, "stream": "species_hidden_0"}]

    built = build(load_config(distilling(held, teacher, loss=terms)))
    before = {name: value.clone() for name, value in built.module.learner.model.named_parameters()}

    run(built)

    after = dict(built.module.learner.model.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("neck.")), (
        "nothing beneath the held-still head moved, so there was no run for these columns to be read from"
    )
    assert all(torch.equal(before[name], after[name]) for name in before if name.startswith(HEAD)), (
        "the held-still head moved"
    )
    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert ALIGNED in columns
    assert HIDDEN in columns


def test_a_term_over_the_middle_of_a_neck_reaches_the_report_by_the_name_the_position_gave_it(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """The one level a distilled pair shares that neither backbone offers: both necks are declared alike,
    so the width between what they read and what they publish is the same width in both, while two
    families' own middles are not comparable at all.

    Published before the norm and before the nonlinearity after it, so this column measures a distance
    that carries scale; what that is worth is the term's weight to say.
    """
    shared = narrowed(declared, SHARED, hidden_features=[SHARED * 2])
    teacher = teacher_of(shared, tmp_path)
    terms = [*TERMS, {"loss": "mse", "weight": 1.0, "stream": "neck_hidden_0"}]

    built = build(load_config(distilling(shared, teacher, loss=terms)))
    before = {name: value.clone() for name, value in built.module.learner.model.named_parameters()}

    run(built)

    after = dict(built.module.learner.model.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("neck.")), (
        "the neck never moved, so there was no run for this column to be read from"
    )
    with sorted((tmp_path / "recorded").rglob("metrics.csv"))[0].open() as recorded:
        columns = next(iter(csv.reader(recorded)))
    assert MIDDLE in columns


def test_a_term_over_a_stream_no_head_publishes_is_refused_naming_what_this_run_does(
    declared: dict[str, Any], tmp_path: Path
) -> None:
    """A head of two hidden widths publishes two, so a term over a third names a stream nobody publishes
    — refused where every stream refusal is, by a list that carries the head's own streams beside the
    encoding half's, which is what says the two are one namespace to a term rather than two.

    A head that publishes rather than the `linear` that publishes nothing, because the list is half of
    what is asserted here and a `linear` run leaves that half empty; that `linear` declares no such
    capability at all is the unit test's to say.
    """
    shared = answering_through(narrowed(declared, SHARED), {"name": "mlp", "hidden_features": [6, 4]})
    teacher = teacher_of(shared, tmp_path)

    with pytest.raises(ValueError, match=r"species_hidden_2'.*publishes pooled, species_hidden_0, species_hidden_1"):
        run(build(load_config(distilling(shared, teacher, loss=[{"loss": "mse", "stream": "species_hidden_2"}]))))
