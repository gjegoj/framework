"""Cut the tail of a teacher's head out of its run checkpoint, as a file a student's head can load.

A run that continues part of a larger network's head declares the tail of it — ``head:
{name: mlp, hidden_features: [128, 64], checkpoint_path: ...}`` — and that file has to hold
exactly that tail, under the names the student's own head uses. Nothing in the framework cuts
it: what to keep is a decision about two architectures, and the run being distilled into has
only ever seen one of them.

The cut is not a prefix strip, which is the whole reason this exists. An ``mlp`` head numbers
its layers through the activations between them, so a head of four projections is ``layers.0,
2, 4, 6`` and its last three are ``layers.2, 4, 6`` — numbers the student's own head, which is
``layers.0, 2, 4``, does not have. This renumbers by building that head and filling it, which
is also what checks the cut: ``load_state_dict`` refuses anything that does not fit, and it is
called before a file is written, so a mistyped width is answered here rather than an hour later
where a run is assembled.

Only ``mlp`` is cut. A head of another class numbers itself another way and would need its own
reading of what a tail even is; there is no second one to serve yet, and a guess would be one.

Run: uv run python scripts/cut_head_tail.py runs/teacher/checkpoints/best.ckpt weights/tail.ckpt \
         --task species --in-features 1280 --out-features 8 --hidden 128 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from src.models.heads import Mlp
from src.training.checkpoints import model_weights


def projections(head: dict[str, Tensor]) -> list[str]:
    """The layer paths a head holds, ordered as they are read through rather than as text sorts them."""
    return sorted({name.rsplit(".", 1)[0] for name in head}, key=lambda path: int(path.rsplit(".", 1)[-1]))


def main() -> None:
    parse = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("checkpoint", type=Path, help="the teacher's run checkpoint")
    parse.add_argument("output", type=Path, help="where to write the tail")
    parse.add_argument("--task", required=True, help="which task's head to cut, as the teacher's run named it")
    parse.add_argument("--in-features", type=int, required=True, help="the width the student's backbone publishes")
    parse.add_argument("--out-features", type=int, required=True, help="how many classes the task declares")
    parse.add_argument("--hidden", type=int, nargs="+", required=True, help="the student's own `hidden_features`")
    written = parse.parse_args()

    prefix = f"heads.{written.task}."
    held = model_weights(str(written.checkpoint))
    taught = {name.removeprefix(prefix): value for name, value in held.items() if name.startswith(prefix)}
    if not taught:
        heads = sorted({name.split(".")[1] for name in held if name.startswith("heads.")})
        raise SystemExit(f"{written.checkpoint} holds no head for {written.task!r}; it holds {', '.join(heads)}.")

    student = Mlp(written.in_features, written.out_features, hidden_features=written.hidden)
    wanted, offered = projections(student.state_dict()), projections(taught)
    if len(offered) < len(wanted):
        raise SystemExit(
            f"The head in {written.checkpoint} has {len(offered)} layers and the tail asked for has "
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

    written.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), written.output)
    reads = [one.in_features for one in student.modules() if isinstance(one, nn.Linear)]
    widths = " → ".join(str(width) for width in [*reads, written.out_features])
    print(f"Cut {', '.join(cut)} of {len(offered)} layers into {written.output} — {widths}.")


if __name__ == "__main__":
    main()
