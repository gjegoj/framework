"""Take a teacher's head out of its run checkpoint, as a file a student's head can load.

A run carrying a head trained elsewhere declares where it came from — ``head: {name: linear,
checkpoint_path: ...}`` — and that file has to hold exactly that head, under the names the
student's own uses. Nothing in the framework cuts it out: which part to keep is a decision
about two architectures, and the run being distilled into has only ever seen one of them.

**The whole head**, where the student carries all of it, which is what two networks brought to
one width by a ``projector`` do. Written by name and nothing else::

    uv run python -m scripts.cut_head_tail runs/teacher/checkpoints/best.ckpt weights/head.ckpt \
        --task species

**Its tail**, where the student's backbone publishes the width a stack passes through partway
along. That is not a prefix strip, which is the whole reason this exists: an ``mlp`` numbers its
layers through the activations between them, so a head of four projections is ``layers.0, 2, 4,
6`` and its last three are ``layers.2, 4, 6`` — numbers the student's own head, which is
``layers.0, 2, 4``, does not have. The widths say which head that is, and building it is also
what checks the cut: ``load_state_dict`` refuses anything that does not fit, and it is called
before a file is written, so a mistyped width is answered here rather than an hour later where
a run is assembled::

    uv run python -m scripts.cut_head_tail runs/teacher/checkpoints/best.ckpt weights/tail.ckpt \
        --task species --in-features 1280 --out-features 8 --hidden 128 64

Only ``mlp`` is cut *into*: a tail is the last layers of a numbered stack, and a head of another class
numbers itself another way and would need its own reading of what a tail even is. Taken whole, a head
of any class is copied as it stands — nothing is cut, and the names it holds are the student's own.


As a module because this one imports the framework, which ``prepare_pet.py`` beside it does not:
run by path, the interpreter puts ``scripts/`` on the import path and the repository root nowhere,
so ``src`` is not there to find. ``-m`` puts the working directory there instead, which is where
``src`` is — the same reason ``uv run main.py`` needs nothing, its file being at the root already.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from src.models.heads import Mlp
from src.training.checkpoints import model_weights


def projections(head: dict[str, Tensor]) -> list[str]:
    """The distinct layer paths a head holds, however many parameters each of them writes."""
    return sorted({name.rsplit(".", 1)[0] for name in head})


def stacked(head: dict[str, Tensor]) -> list[str]:
    """Those paths in the order they are read through, refused by name where the head is no stack.

    A stack numbers its layers — ``mlp`` writes ``layers.0``, ``layers.2`` — and the number is the
    order, because text would put ``layers.10`` before ``layers.2``. What makes a head cuttable is
    that numbering and not how many paths it holds: ``linear`` writes one unnumbered ``projection``
    and ``cosine`` writes ``prototypes`` beside it, and neither is a layer of a stack. Such a head has
    no tail short of itself, which is the whole-head cut, so the refusal names that way out.
    """
    return sorted(projections(head), key=_read_through)


def _read_through(path: str) -> int:
    """Where a layer sits in the stack it belongs to, refused by name where its path does not say."""
    numbered = path.rsplit(".", 1)[-1]
    if not numbered.isdecimal():
        raise SystemExit(
            f"{path!r} is not a numbered layer, and a tail is the last layers of a stack such as `mlp` "
            f"writes. A head that numbers nothing is one whole thing: leave the widths out and it is "
            f"taken whole."
        )
    return int(numbered)


def main() -> None:
    parse = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("checkpoint", type=Path, help="the teacher's run checkpoint")
    parse.add_argument("output", type=Path, help="where to write the tail")
    parse.add_argument("--task", required=True, help="which task's head to cut, as the teacher's run named it")
    parse.add_argument("--in-features", type=int, help="the width the student's backbone publishes")
    parse.add_argument("--out-features", type=int, help="how many classes the task declares")
    parse.add_argument("--hidden", type=int, nargs="+", help="the student's own `hidden_features`")
    asked = parse.parse_args()

    prefix = f"heads.{asked.task}."
    held = model_weights(str(asked.checkpoint))
    taught = {name.removeprefix(prefix): value for name, value in held.items() if name.startswith(prefix)}
    if not taught:
        heads = sorted({name.split(".")[1] for name in held if name.startswith("heads.")})
        raise SystemExit(f"{asked.checkpoint} holds no head for {asked.task!r}; it holds {', '.join(heads)}.")

    widths = {"--in-features": asked.in_features, "--out-features": asked.out_features, "--hidden": asked.hidden}
    if not any(value is not None for value in widths.values()):
        # The whole head, which is what a student carrying all of its teacher's declares. Nothing to
        # renumber and nothing to check: these are the very names and widths the student's head will
        # build, since it is the same head — whatever class it is, since none of it is being cut into.
        _written(asked.output, taught)
        print(f"Took the whole of {asked.task!r}'s head into {asked.output} — {', '.join(projections(taught))}.")
        return
    missing = sorted(name for name, value in widths.items() if value is None)
    if missing:
        raise SystemExit(
            f"A tail is the last layers of a head, and which layers those are is said by the head the "
            f"student builds: {', '.join(missing)} not given. Give all three, or none of them for the "
            f"whole head."
        )

    student = Mlp(asked.in_features, asked.out_features, hidden_features=asked.hidden)
    wanted, offered = stacked(student.state_dict()), stacked(taught)

    if len(offered) < len(wanted):
        raise SystemExit(
            f"The head in {asked.checkpoint} has {len(offered)} layers and the tail asked for has "
            f"{len(wanted)}; a tail is shorter than the head it is cut from."
        )
    cut = offered[-len(wanted) :]
    student.load_state_dict(
        {
            f"{into}.{part}": taught[f"{outof}.{part}"]
            for into, outof in zip(wanted, cut, strict=True)
            for part in ("weight", "bias")
        }
    )

    _written(asked.output, student.state_dict())
    reads = [one.in_features for one in student.modules() if isinstance(one, nn.Linear)]
    through = " → ".join(str(width) for width in [*reads, asked.out_features])
    print(f"Cut {', '.join(cut)} of {len(offered)} layers into {asked.output} — {through}.")


def _written(output: Path, head: dict[str, Tensor]) -> None:
    """The file, and the directory a reader named for it; one home, because two paths end here."""
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, output)


if __name__ == "__main__":
    main()
