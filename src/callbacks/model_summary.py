"""The model summary's Name column as a module tree, so the hierarchy reads at a glance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, override

from lightning.pytorch.callbacks import RichModelSummary

from src.callbacks.registry import callback_registry
from src.console import HEADER_STYLE

BRANCH, LAST, THROUGH, PAST = "├─ ", "└─ ", "│  ", "   "
"""The four pieces a tree is drawn from: a child, the last child, a line passing under one, and none."""


def tree_names(paths: list[str]) -> list[str]:
    """Ordered dotted module paths as tree-connected leaf names, in the order they arrived.

    ``["model", "model.backbone", "model.backbone.encoder", "model.heads"]`` becomes
    ``["model", "├─ backbone", "│  └─ encoder", "└─ heads"]``. The order is kept because every other
    column of the summary is a list parallel to this one: a row moved here takes another's numbers.

    A pure function, so the tree is testable without a model to summarise.
    """
    children: dict[str | None, list[str]] = defaultdict(list)
    for path in paths:
        children[path.rsplit(".", 1)[0] if "." in path else None].append(path)

    drawn: dict[str, str] = {}

    def descend(parent: str | None, prefix: str) -> None:
        siblings = children.get(parent, [])
        for index, path in enumerate(siblings):
            last, leaf = index == len(siblings) - 1, path.rsplit(".", 1)[-1]
            drawn[path] = leaf if parent is None else f"{prefix}{LAST if last else BRANCH}{leaf}"
            descend(path, prefix if parent is None else prefix + (PAST if last else THROUGH))

    descend(None, "")
    return [drawn[path] for path in paths]


@callback_registry.register("model_summary")
class TreeModelSummary(RichModelSummary):
    """Lightning's rich summary with its Name column drawn as the tree those names describe.

    Lightning prints flat dotted paths; the same rows as a tree say where a module sits without being
    read character by character — and the paths a config freezes or loads by (``model.heads.<task>``)
    are the very ones shown. Only that column is touched: everything else is passed on as it came, so
    a column Lightning adds later needs no edit here.
    """

    @staticmethod
    @override
    def summarize(summary_data: list[tuple[str, list[str]]], *rest: Any, **options: Any) -> None:
        treed = [(header, tree_names(values) if header == "Name" else values) for header, values in summary_data]
        # The same header the tables under this one wear. Set rather than left to the library's own
        # default, which is this value today: that is what makes the three one statement instead of two.
        options.setdefault("header_style", HEADER_STYLE)
        RichModelSummary.summarize(treed, *rest, **options)
