"""TreeModelSummary: a ``RichModelSummary`` that renders the module hierarchy as a tree.

Lightning's ``RichModelSummary`` prints a flat table — one row per module, the full
dotted name, and a leading index column. This subclass reuses all of Lightning's data
(recursive param counts, train/eval modes, FLOPs, the totals) and changes only the
*presentation*: the ``Name`` column becomes a box-drawing tree of leaf names and the
index column is dropped. Columns, footer, rank-zero gating and ``max_depth`` are
inherited unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import get_formatted_model_size, get_human_readable_count
from rich import get_console
from rich.table import Table

# Lightning's leading index column header; we drop it — the tree connectors carry the structure.
_INDEX_HEADER = " "
_RIGHT_JUSTIFIED = frozenset({"Params", "FLOPs", "Params per Device", "In sizes", "Out sizes"})


def tree_names(names: list[str]) -> list[str]:
    """Turn ordered dotted module paths into tree-connector-prefixed leaf labels.

    ``["model", "model.backbone", "model.backbone.encoder", "model.heads"]`` becomes
    ``["model", "├─ backbone", "│  └─ encoder", "└─ heads"]`` (input order preserved).
    A pure function, so the tree logic is unit-testable without Lightning.

    Parameters:
        names (list[str]): Dotted module paths in summary (pre-order) order.

    Returns:
        list[str]: One label per input name, prefixed with tree connectors.
    """
    children: dict[str | None, list[str]] = defaultdict(list)
    for name in names:
        parent = name.rsplit(".", 1)[0] if "." in name else None
        children[parent].append(name)

    label: dict[str, str] = {}

    def render(parent: str | None, prefix: str) -> None:
        siblings = children.get(parent, [])
        for index, name in enumerate(siblings):
            is_last = index == len(siblings) - 1
            leaf = name.rsplit(".", 1)[-1]
            if parent is None:
                label[name] = leaf
                render(name, "")
            else:
                label[name] = f"{prefix}{'└─ ' if is_last else '├─ '}{leaf}"
                render(name, prefix + ("   " if is_last else "│  "))

    render(None, "")
    return [label[name] for name in names]


class TreeModelSummary(RichModelSummary):
    """``RichModelSummary`` with the ``Name`` column rendered as a module tree."""

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: Any,
    ) -> None:
        # Drop the index column; replace dotted names with tree-prefixed leaf labels.
        body = [
            (header, tree_names(values) if header == "Name" else values)
            for header, values in summary_data
            if header != _INDEX_HEADER
        ]

        table = Table(header_style=summarize_kwargs.get("header_style", "bold magenta"))
        for header, _ in body:
            table.add_column(
                header, justify="right" if header in _RIGHT_JUSTIFIED else "left", no_wrap=header == "Name"
            )
        for row in zip(*(values for _, values in body), strict=True):
            table.add_row(*row)

        console = get_console()
        console.print(table)
        console.print(_footer(total_parameters, trainable_parameters, model_size, total_training_modes, total_flops))


def _footer(
    total_parameters: int,
    trainable_parameters: int,
    model_size: float,
    total_training_modes: dict[str, int],
    total_flops: int,
) -> Table:
    """Build the totals footer grid, matching ``RichModelSummary``'s layout."""
    trainable = get_human_readable_count(int(trainable_parameters))
    non_trainable = get_human_readable_count(int(total_parameters - trainable_parameters))
    total = get_human_readable_count(int(total_parameters))
    size = get_formatted_model_size(model_size)
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column()
    grid.add_row(f"[bold]Trainable params[/]: {trainable:<10}")
    grid.add_row(f"[bold]Non-trainable params[/]: {non_trainable:<10}")
    grid.add_row(f"[bold]Total params[/]: {total:<10}")
    grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {size:<10}")
    grid.add_row(f"[bold]Modules in train mode[/]: {total_training_modes['train']}")
    grid.add_row(f"[bold]Modules in eval mode[/]: {total_training_modes['eval']}")
    grid.add_row(f"[bold]Total FLOPs[/]: {get_human_readable_count(total_flops)}")
    return grid
